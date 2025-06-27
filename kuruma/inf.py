# -*- coding: utf-8 -*-
"""
Fast-SCNN TUSimple Lane Segmentation - Ascend Inference Script (High-Level API)
使用 ais_bench 库在昇腾设备上进行OM模型推理，代码更简洁。
"""
import os
import cv2
import time
import numpy as np
from ais_bench.infer.interface import InferSession

# --- 可配置常量 ---
# NPU设备ID
DEVICE_ID = 0
# OM模型路径
MODEL_PATH = "./weights/fast_scnn_tusimple_bs1.om"
# 测试图片输入目录
INPUT_DIR = "./test_images/"
# 推理结果输出目录
OUTPUT_DIR = "./inference_results_ais_bench/"
# 模型要求的输入尺寸 (根据atc命令)
MODEL_WIDTH = 1024
MODEL_HEIGHT = 768
# --------------------

def preprocess(image_path):
    """
    读取并预处理图像，与之前的脚本完全相同。
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    original_height, original_width = img_bgr.shape[:2]

    # 1. Resize
    img_resized = cv2.resize(img_bgr, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_LINEAR)
    # 2. BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    # 3. ToTensor & Normalize
    img_tensor = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_tensor - mean) / std
    # 4. HWC to NCHW
    img_transposed = img_normalized.transpose(2, 0, 1)
    # 5. Add batch dimension and ensure C-contiguous
    input_data = np.ascontiguousarray(img_transposed[np.newaxis, :, :, :], dtype=np.float32)

    return input_data, original_width, original_height

def postprocess(output_tensor, original_width, original_height):
    """
    处理推理结果并可视化，与之前的脚本完全相同。
    """
    # 1. Argmax
    pred_mask = np.argmax(output_tensor, axis=1).squeeze()
    # 2. 转换为可视化图像 (0=背景, 255=车道线)
    vis_mask = (pred_mask * 255).astype(np.uint8)
    # 3. 缩放回原始尺寸
    vis_mask_resized = cv2.resize(vis_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    return vis_mask_resized

def main():
    """主函数"""
    try:
        # 1. 初始化模型会话 (ais_bench 会自动处理ACL初始化和资源管理)
        print("Initializing inference session with ais_bench...")
        model = InferSession(DEVICE_ID, MODEL_PATH)
        print("Model loaded successfully.")

        # 确保输出目录存在
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"Created output directory: {OUTPUT_DIR}")

        # 2. 获取测试图片列表
        test_images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not test_images:
            print(f"No test images found in {INPUT_DIR}")
            return

        # 3. 循环推理
        total_time = 0
        for image_name in test_images:
            image_path = os.path.join(INPUT_DIR, image_name)
            print(f"\n--- Processing {image_path} ---")

            try:
                # 预处理
                input_data, orig_w, orig_h = preprocess(image_path)

                # 推理 (ais_bench将复杂的步骤简化为一行调用)
                start_time = time.time()
                # model.infer() 需要一个包含所有输入的列表
                outputs = model.infer([input_data])
                inference_time = time.time() - start_time
                total_time += inference_time
                print(f"Inference successful. Time taken: {inference_time*1000:.2f} ms")

                # 后处理 (outputs 是一个列表，我们取第一个)
                final_mask = postprocess(outputs[0], orig_w, orig_h)

                # 保存结果
                save_path = os.path.join(OUTPUT_DIR, f"result_{image_name}")
                cv2.imwrite(save_path, final_mask)
                print(f"Result saved to {save_path}")

            except Exception as e:
                print(f"An error occurred while processing {image_name}: {e}")

        avg_time = total_time / len(test_images) if test_images else 0
        print(f"\n✅ All images processed. Average inference time: {avg_time*1000:.2f} ms")

    except Exception as e:
        print(f"A critical error occurred: {e}")
    finally:
        # ais_bench 的 InferSession 对象在销毁时会自动释放所有ACL资源
        # 无需手动调用 acl.finalize() 等清理函数
        print("\nProgram finished. Resources are managed automatically by ais_bench.")


if __name__ == "__main__":
    main()