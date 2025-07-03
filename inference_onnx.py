import onnxruntime as ort
import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt

def main(args):
    """主函数，用于加载模型、图像并进行推理和可视化"""
    # 1. 检查输入文件是否存在
    if not os.path.exists(args.model):
        print(f"❌ 错误: ONNX模型文件未找到: {args.model}")
        return
    if not os.path.exists(args.image):
        print(f"❌ 错误: 图像文件未找到: {args.image}")
        return

    print("🚀 开始ONNX模型推理...")
    print(f"   - 模型: {args.model}")
    print(f"   - 图像: {args.image}")

    # 2. 创建ONNX推理会话
    try:
        session = ort.InferenceSession(args.model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print(f"✅ ONNX会话创建成功，使用: {session.get_providers()[0]}")
    except Exception as e:
        print(f"❌ 创建ONNX会话失败: {e}")
        return

    # 3. 加载并预处理图像
    # 模型输入尺寸为 360x640
    input_shape = session.get_inputs()[0].shape # (1, 3, 360, 640)
    input_height, input_width = input_shape[2], input_shape[3]

    original_image = cv2.imread(args.image)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # 准备输入张量
    # 注意：模型是端到端的，它自己处理resize和归一化，我们只需要送入原始的uint8图像数据
    # 但ONNX Runtime需要float32输入，所以我们进行类型转换
    input_tensor = cv2.resize(original_image, (input_width, input_height))
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32) # Add batch dim and convert to float32

    print(f"✅ 图像加载并预处理完成，输入尺寸: {input_tensor.shape}")

    # 4. 执行推理
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print("🧠 正在执行推理...")
    result = session.run([output_name], {input_name: input_tensor})[0]
    print("✅ 推理完成!")
    print(f"   - 输出尺寸: {result.shape}") # (1, 2, 360, 640)

    # 5. 后处理：获取分割掩码
    # result是softmax的输出概率，我们需要找到哪个类别概率最大
    predicted_mask = np.argmax(result, axis=1)[0] # (360, 640)
    
    # 创建一个彩色的叠加层
    # 假设类别1是可行驶区域，我们将其设为绿色
    overlay = np.zeros((input_height, input_width, 3), dtype=np.uint8)
    overlay[predicted_mask == 1] = [0, 255, 0] # BGR for OpenCV

    # 6. 可视化结果
    print("🎨 正在生成可视化结果...")
    
    # 将原始图像（已缩放）和叠加层混合
    # 我们需要一个与叠加层相同尺寸的原始图像
    resized_original_image = cv2.resize(original_image_rgb, (input_width, input_height))
    
    # 将RGB转回BGR以供cv2使用
    resized_original_image_bgr = cv2.cvtColor(resized_original_image, cv2.COLOR_RGB2BGR)
    
    # 混合图像
    blended_image = cv2.addWeighted(resized_original_image_bgr, 1, overlay, 0.5, 0)

    # 创建一个大的画布来并排显示
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像
    axes[0].imshow(resized_original_image)
    axes[0].set_title('Original Image (Resized)')
    axes[0].axis('off')

    # 预测的掩码
    axes[1].imshow(predicted_mask, cmap='gray')
    axes[1].set_title('Predicted Mask (Class 1 is white)')
    axes[1].axis('off')

    # 叠加效果
    axes[2].imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Prediction Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    output_filename = os.path.join(os.path.dirname(args.model), "inference_result.png")
    plt.savefig(output_filename, dpi=150)
    print(f"✅ 结果已保存到: {output_filename}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ONNX inference for Fast-SCNN and visualize results.")
    parser.add_argument('--model', type=str, 
                        default='./weights/fast_scnn_custom_e2e_360x640_simplified.onnx',
                        help='Path to the simplified ONNX model.')
    parser.add_argument('--image', type=str, 
                        default='./data/custom/images/0a0a0a0a-64a53900.jpg',
                        help='Path to the input image.')
    
    # 自动查找一个测试图片
    if not os.path.exists(parser.parse_args().image):
        print(f"警告: 默认图片 '{parser.parse_args().image}' 不存在。")
        image_dir = './data/custom/images'
        if os.path.exists(image_dir) and len(os.listdir(image_dir)) > 0:
            found_image = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))][0]
            parser.set_defaults(image=os.path.join(image_dir, found_image))
            print(f"将使用找到的第一张图片进行测试: {parser.parse_args().image}")
        else:
            print("错误: 在 './data/custom/images' 目录下找不到任何图片文件。请手动指定 --image 参数。")
            exit()

    args = parser.parse_args()
    main(args)
