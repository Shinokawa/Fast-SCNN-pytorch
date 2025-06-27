# -*- coding: utf-8 -*-
"""
Fast-SCNN TUSimple Lane Segmentation - Ascend Inference Script
使用pyacl在昇腾设备上进行OM模型推理。
"""
import os
import cv2
import numpy as np
import acl

# 定义常量
# NPU设备ID
DEVICE_ID = 0
# OM模型路径
MODEL_PATH = "../weights/fast_scnn_tusimple_bs1.om"
# 测试图片输入目录
INPUT_DIR = "./test_images/"
# 推理结果输出目录
OUTPUT_DIR = "./output/"
# 模型要求的输入尺寸
MODEL_WIDTH = 1024
MODEL_HEIGHT = 768

def init_acl(device_id=0):
    """初始化ACL资源"""
    print("Initializing ACL...")
    ret = acl.init()
    if ret != 0:
        raise RuntimeError(f"ACL init failed. Error code: {ret}")
    
    ret = acl.rt.set_device(device_id)
    if ret != 0:
        raise RuntimeError(f"Failed to set device {device_id}. Error code: {ret}")
    
    context, ret = acl.rt.create_context(device_id)
    if ret != 0:
        raise RuntimeError(f"Failed to create context. Error code: {ret}")
    
    print("ACL initialized successfully.")
    return context

def load_model(model_path):
    """加载OM模型"""
    print(f"Loading model from {model_path}...")
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    model_id, ret = acl.mdl.load_from_file(model_path)
    if ret != 0:
        raise RuntimeError(f"Failed to load model. Error code: {ret}")
    
    model_desc = acl.mdl.create_desc()
    acl.mdl.get_desc(model_desc, model_id)
    input_count = acl.mdl.get_num_inputs(model_desc)
    output_count = acl.mdl.get_num_outputs(model_desc)
    print(f"Model loaded. Inputs: {input_count}, Outputs: {output_count}")
    
    return model_id, model_desc

def preprocess_image(image_path):
    """读取并预处理图像"""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # 记录原始尺寸用于后处理
    original_height, original_width = img_bgr.shape[:2]

    # 1. Resize
    img_resized = cv2.resize(img_bgr, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    # 2. BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # 3. Normalize
    # 与PyTorch transforms.Normalize([.485, .456, .406], [.229, .224, .225])一致
    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    std = np.array([0.229, 0.224, 0.225]) * 255.0
    img_normalized = (img_rgb.astype(np.float32) - mean) / std
    
    # 4. HWC to NCHW
    img_transposed = img_normalized.transpose(2, 0, 1)
    
    # 5. Add batch dimension and ensure C-contiguous
    input_data = np.ascontiguousarray(img_transposed[np.newaxis, :, :, :])
    
    return input_data, original_width, original_height

def get_numpy_dtype(acl_dtype):
    """将ACL数据类型枚举映射到Numpy数据类型"""
    ACL_TO_NUMPY_DTYPE = {
        0: np.float32,   # ACL_FLOAT
        1: np.float16,   # ACL_FLOAT16
        2: np.uint8,     # ACL_UINT8
        3: np.int8,      # ACL_INT8
        6: np.int32,     # ACL_INT32
    }
    return ACL_TO_NUMPY_DTYPE.get(acl_dtype)

def run_inference(model_id, model_desc, input_data):
    """执行模型推理"""
    # 1. 创建输入数据集
    input_dataset = acl.mdl.create_dataset()
    input_size = input_data.nbytes
    
    # 2. 申请Device上的内存
    input_ptr, ret = acl.rt.malloc(input_size, 0) # 使用整数 0 替代内存分配常量
    if ret != 0:
        raise RuntimeError("Failed to allocate device memory for input.")
        
    # 3. 将数据从Host拷贝到Device
    ret = acl.rt.memcpy(input_ptr, input_size, input_data.ctypes.data, input_size, 1) # 使用整数 1 替代 ACL_MEMCPY_HOST_TO_DEVICE
    if ret != 0:
        # 释放已申请的内存
        acl.rt.free(input_ptr)
        raise RuntimeError("Failed to copy input data to device.")

    # 4. 创建数据缓冲并添加到数据集中
    data_buffer = acl.create_data_buffer(input_ptr, input_size)
    _, ret = acl.mdl.add_dataset_buffer(input_dataset, data_buffer)
    if ret != 0:
        # 清理资源
        acl.rt.free(input_ptr)
        acl.destroy_data_buffer(data_buffer)
        raise RuntimeError("Failed to add input buffer to dataset.")

    # 5. 创建输出数据集
    output_dataset = acl.mdl.create_dataset()
    output_count = acl.mdl.get_num_outputs(model_desc)
    output_pointers = []
    for i in range(output_count):
        output_size = acl.mdl.get_output_size_by_index(model_desc, i)
        output_ptr, ret = acl.rt.malloc(output_size, 0) # 使用整数 0 替代内存分配常量
        if ret != 0:
            raise RuntimeError(f"Failed to allocate device memory for output {i}.")
        output_pointers.append(output_ptr)
        output_buffer = acl.create_data_buffer(output_ptr, output_size)
        _, ret = acl.mdl.add_dataset_buffer(output_dataset, output_buffer)
        if ret != 0:
            raise RuntimeError(f"Failed to add output buffer to dataset for output {i}.")

    # 6. 执行模型
    ret = acl.mdl.execute(model_id, input_dataset, output_dataset)
    if ret != 0:
        raise RuntimeError(f"Model execution failed. Error code: {ret}")

    # 7. 获取输出数据
    output_tensors = []
    for i in range(output_count):
        buffer_ptr = acl.get_data_buffer_addr(acl.mdl.get_dataset_buffer(output_dataset, i))
        buffer_size = acl.get_data_buffer_size(acl.mdl.get_dataset_buffer(output_dataset, i))
        
        # 动态获取输出的shape和dtype
        output_shape = acl.mdl.get_output_dims(model_desc, i)[0]['dims']
        acl_dtype_enum = acl.mdl.get_output_data_type(model_desc, i)
        output_dtype = get_numpy_dtype(acl_dtype_enum)

        print(f"--> Output {i}: Shape={output_shape}, ACL DType={acl_dtype_enum}, Numpy DType={output_dtype}")

        if output_dtype is None:
            raise TypeError(f"Unsupported ACL data type: {acl_dtype_enum} for output {i}")

        # 创建一个正确类型的空的numpy数组用于接收数据
        host_buffer = np.empty(output_shape, dtype=output_dtype)
        
        # 将数据从Device拷贝回Host
        ret = acl.rt.memcpy(host_buffer.ctypes.data, host_buffer.nbytes, buffer_ptr, buffer_size, 2) # 使用整数 2 替代 ACL_MEMCPY_DEVICE_TO_HOST
        if ret != 0:
            raise RuntimeError(f"Failed to copy output data from device for output {i}.")
            
        output_tensors.append(host_buffer)

    # 8. 清理资源
    acl.rt.free(input_ptr)
    acl.destroy_data_buffer(data_buffer)
    acl.mdl.destroy_dataset(input_dataset)
    for i in range(output_count):
        buffer_to_destroy = acl.mdl.get_dataset_buffer(output_dataset, i)
        acl.destroy_data_buffer(buffer_to_destroy)
        acl.rt.free(output_pointers[i])
    acl.mdl.destroy_dataset(output_dataset)
    
    return output_tensors

def postprocess_result(output_tensor, original_width, original_height):
    """
    处理推理结果并可视化。
    [诊断模式] 本版本通过可视化logit差值来诊断模型输出问题。
    """
    # 确保数据类型为float32，以便进行数学运算
    logits = output_tensor.astype(np.float32)
    
    # 提取背景和车道线的logit
    # logit shape: (1, 2, H, W)
    background_logits = logits[0, 0, :, :]
    lane_logits = logits[0, 1, :, :]
    
    # 计算车道线logit与背景logit的差值。正值表示车道线可能性更高。
    logit_difference = lane_logits - background_logits
    
    # 将差值归一化到0-255范围以便可视化。
    # 车道线区域会更亮，背景区域会更暗。
    vis_mask = cv2.normalize(logit_difference, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 缩放回原始尺寸
    vis_mask_resized = cv2.resize(vis_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    
    return vis_mask_resized

def main():
    """主函数"""
    context = None
    model_id = None
    try:
        # 1. 初始化
        context = init_acl(DEVICE_ID)
        model_id, model_desc = load_model(MODEL_PATH)
        
        # 确保输出目录存在
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        # 2. 获取测试图片列表
        test_images = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.jpg', '.png'))]
        if not test_images:
            print(f"No test images found in {INPUT_DIR}")
            return

        # 3. 循环推理
        for image_name in test_images:
            image_path = os.path.join(INPUT_DIR, image_name)
            print(f"\n--- Processing {image_path} ---")
            
            try:
                # 预处理
                input_data, orig_w, orig_h = preprocess_image(image_path)
                
                # 推理
                output_tensors = run_inference(model_id, model_desc, input_data)
                
                # 后处理
                # 假设我们只关心第一个输出
                final_mask = postprocess_result(output_tensors[0], orig_w, orig_h)
                
                # 保存结果
                save_path = os.path.join(OUTPUT_DIR, f"result_{image_name}")
                cv2.imwrite(save_path, final_mask)
                print(f"Result saved to {save_path}")
            except Exception as e:
                print(f"Error processing {image_name}: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 4. 清理
        print("\nCleaning up resources...")
        if model_id is not None:
            acl.mdl.unload(model_id)
        if 'model_desc' in locals() and model_desc is not None:
            acl.mdl.destroy_desc(model_desc)
        if context is not None:
            acl.rt.destroy_context(context)
        acl.rt.reset_device(DEVICE_ID)
        acl.finalize()
        print("Done.")

if __name__ == "__main__":
    main()
