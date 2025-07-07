#!/usr/bin/env python3
"""
多模型Atlas推理测试脚本

验证单进程多线程多Stream架构下的两个模型并发运行
"""

import os
import time
import threading
import cv2
import numpy as np
from pathlib import Path

# 导入核心模块
from core.atlas_session_manager import get_atlas_session
from core.logging_config import setup_unified_logging, get_module_logger
from vision.traffic_light_detection import create_traffic_light_detector

def test_lane_segmentation_model():
    """测试车道线分割模型"""
    logger = get_module_logger("lane_test")
    
    try:
        # 导入推理模块
        from core.inference import AtlasInferSession
        
        # 模型路径
        model_path = "./weights/fast_scnn_custom_e2e_360x640_fp16_fixed_simp.om"
        
        if not os.path.exists(model_path):
            logger.error(f"❌ 车道线分割模型不存在: {model_path}")
            return False
        
        logger.info("🛣️ 开始测试车道线分割模型...")
        
        # 创建推理会话
        inference_session = AtlasInferSession(
            device_id=0,
            model_path=model_path,
            model_name="lane_segmentation_test"
        )
        
        # 创建虚拟输入数据
        input_data = np.random.randn(1, 3, 360, 640).astype(np.float16)
        
        # 执行推理测试
        for i in range(5):
            start_time = time.time()
            outputs = inference_session.infer([input_data])
            inference_time = (time.time() - start_time) * 1000
            
            logger.info(f"🛣️ 车道线分割推理 {i+1}/5: {inference_time:.2f}ms, 输出形状: {outputs[0].shape}")
            time.sleep(0.1)  # 短暂休眠
        
        logger.info("✅ 车道线分割模型测试完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 车道线分割模型测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_traffic_light_model():
    """测试交通灯检测模型"""
    logger = get_module_logger("traffic_test")
    
    try:
        logger.info("🚦 开始测试交通灯检测模型...")
        
        # 创建交通灯检测器
        detector = create_traffic_light_detector()
        
        if detector is None:
            logger.error("❌ 交通灯检测器创建失败")
            return False
        
        # 创建虚拟图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 执行检测测试
        for i in range(5):
            start_time = time.time()
            result = detector.detect_traffic_light(test_image)
            detection_time = (time.time() - start_time) * 1000
            
            logger.info(f"🚦 交通灯检测 {i+1}/5: {detection_time:.2f}ms, 状态: {result['status']}")
            time.sleep(0.1)  # 短暂休眠
        
        logger.info("✅ 交通灯检测模型测试完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 交通灯检测模型测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_concurrent_inference():
    """测试并发推理"""
    logger = get_module_logger("concurrent_test")
    
    logger.info("🎯 开始并发推理测试...")
    
    # 创建线程列表
    threads = []
    results = {}
    
    def lane_worker():
        results['lane'] = test_lane_segmentation_model()
    
    def traffic_worker():
        results['traffic'] = test_traffic_light_model()
    
    # 创建并启动线程
    lane_thread = threading.Thread(target=lane_worker, name="LaneSegmentationThread")
    traffic_thread = threading.Thread(target=traffic_worker, name="TrafficLightThread")
    
    threads.extend([lane_thread, traffic_thread])
    
    # 启动所有线程
    start_time = time.time()
    for thread in threads:
        thread.start()
        logger.info(f"🚀 启动线程: {thread.name}")
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
        logger.info(f"✅ 线程完成: {thread.name}")
    
    total_time = time.time() - start_time
    
    # 检查结果
    all_success = all(results.values())
    
    if all_success:
        logger.info(f"🎉 并发推理测试成功！总耗时: {total_time:.2f}秒")
        logger.info(f"📊 结果: 车道线分割: {'成功' if results.get('lane') else '失败'}, "
                   f"交通灯检测: {'成功' if results.get('traffic') else '失败'}")
    else:
        logger.error(f"❌ 并发推理测试失败！总耗时: {total_time:.2f}秒")
        logger.error(f"📊 结果: 车道线分割: {'成功' if results.get('lane') else '失败'}, "
                    f"交通灯检测: {'成功' if results.get('traffic') else '失败'}")
    
    return all_success

def test_atlas_session_manager():
    """测试Atlas会话管理器"""
    logger = get_module_logger("session_test")
    
    logger.info("🔧 开始测试Atlas会话管理器...")
    
    try:
        # 获取会话管理器
        session_manager = get_atlas_session()
        
        # 测试初始化
        if not session_manager.initialize_atlas(0):
            logger.error("❌ Atlas环境初始化失败")
            return False
        
        logger.info("✅ Atlas环境初始化成功")
        
        # 检查状态
        logger.info(f"📊 设备ID: {session_manager.get_device_id()}")
        logger.info(f"📊 是否已初始化: {session_manager.is_initialized()}")
        logger.info(f"📊 已加载模型列表: {session_manager.list_models()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Atlas会话管理器测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主测试函数"""
    # 配置日志
    setup_unified_logging(
        log_file="test_multi_model.log",
        level="INFO",
        console_output=True
    )
    
    logger = get_module_logger(__name__)
    
    logger.info("🧪 开始多模型Atlas推理测试")
    logger.info("=" * 60)
    
    # 测试步骤
    tests = [
        ("Atlas会话管理器", test_atlas_session_manager),
        ("车道线分割模型", test_lane_segmentation_model),
        ("交通灯检测模型", test_traffic_light_model),
        ("并发推理", test_concurrent_inference),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 测试: {test_name}")
        logger.info("-" * 40)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ 测试 {test_name} 出现异常: {e}")
            results[test_name] = False
        
        if results[test_name]:
            logger.info(f"✅ 测试 {test_name} 通过")
        else:
            logger.error(f"❌ 测试 {test_name} 失败")
    
    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("📋 测试总结")
    logger.info("=" * 60)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！多模型架构运行正常")
        return True
    else:
        logger.error("❌ 部分测试失败，请检查配置")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 