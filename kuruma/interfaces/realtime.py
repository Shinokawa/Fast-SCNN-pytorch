#!/usr/bin/env python3
"""
实时推理接口模块 - 摄像头实时推理功能

包含：
- setup_logging: 日志配置功能
- realtime_inference: 实时摄像头推理主函数
- 完整的推理循环和控制集成
"""

import logging
import time
import cv2
import numpy as np
from threading import Lock

# 导入依赖模块
from core.preprocessing import preprocess_matched_resolution, postprocess_matched_resolution
from core.inference import AtlasInferSession
from vision.transform import PerspectiveTransformer
from vision.path_planning import create_control_map
from control.visual_controller import VisualLateralErrorController

# 尝试导入小车控制器
try:
    from car_controller_simple import SimpleCarController
    CAR_CONTROLLER_AVAILABLE = True
    print("✅ 小车控制模块加载成功")
except ImportError:
    print("⚠️ 警告：未找到car_controller_simple模块，串口控制功能不可用")
    CAR_CONTROLLER_AVAILABLE = False

# ---------------------------------------------------------------------------------
# --- 🚀 实时推理模块 (摄像头模式) ---
# ---------------------------------------------------------------------------------

def setup_logging(log_file=None):
    """配置日志系统"""
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

def realtime_inference(model_path, device_id=0, camera_index=0, 
                      camera_width=640, camera_height=360,
                      log_file=None, enable_control=True,
                      steering_gain=1.0, base_speed=10.0, 
                      curvature_damping=0.1, preview_distance=30.0,
                      max_speed=1000.0, min_speed=5.0,
                      enable_web=False, no_gui=False, full_image_bird_eye=True,
                      edge_computing=False, pixels_per_unit=20, margin_ratio=0.1,
                      ema_alpha=0.5, enable_smoothing=True):
    """
    实时摄像头推理模式
    
    参数：
        model_path: OM模型路径
        device_id: Atlas NPU设备ID
        camera_index: 摄像头索引
        camera_width: 摄像头宽度
        camera_height: 摄像头高度
        log_file: 日志文件路径
        enable_control: 是否启用控制算法
        enable_web: 是否启用Web界面数据更新
        no_gui: 是否禁用GUI显示
        其他: 控制参数
    """
    # 由于需要访问Web界面数据，这些需要从外部传入或重新组织
    # 暂时使用局部变量来避免循环依赖
    global car_controller
    car_controller = None
    
    # 模拟web_data和web_data_lock（在实际应用中应该从外部传入）
    web_data = {
        'is_running': False,
        'frame_count': 0,
        'start_time': None,
        'latest_control_map': None,
        'latest_stats': {},
        'control_params': {},
        'params_updated': False,
        'serial_connected': False,
        'car_driving': False,
        'emergency_stop': False,
        'last_control_command': None,
        'control_enabled': False
    }
    web_data_lock = Lock()
    
    # 配置日志
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 启动实时推理系统")
    logger.info(f"📱 模型: {model_path}")
    logger.info(f"💾 日志文件: {log_file}")
    
    # 加载Atlas模型
    logger.info("🧠 加载Atlas NPU模型...")
    model = AtlasInferSession(device_id, model_path)
    logger.info("✅ 模型加载完成")
    
    # 初始化摄像头
    logger.info(f"📷 打开摄像头 {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"❌ 无法打开摄像头 {camera_index}")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # 确认实际参数
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"📷 摄像头参数: {actual_w}x{actual_h} @ {actual_fps}fps")
    
    # 初始化控制器
    if enable_control:
        # 使用当前文件中的控制器类
        controller = VisualLateralErrorController(
            steering_gain=steering_gain,
            base_pwm=int(base_speed),
            curvature_damping=curvature_damping,
            preview_distance=preview_distance,
            max_pwm=int(max_speed),
            min_pwm=int(min_speed),
            ema_alpha=ema_alpha,
            enable_smoothing=enable_smoothing
        )
        logger.info("🚗 控制器初始化完成")
    else:
        controller = None
    
    # 透视变换器
    transformer = PerspectiveTransformer()
    logger.info("🦅 透视变换器初始化完成")
    
    # 初始化Web界面数据
    if enable_web:
        with web_data_lock:
            web_data['is_running'] = True
            web_data['start_time'] = time.time()
            web_data['frame_count'] = 0
        logger.info("🌐 Web界面数据初始化完成")
    
    frame_count = 0
    start_time = time.time()
    total_times = {"preprocess": 0, "inference": 0, "postprocess": 0, "transform": 0, "control": 0}
    
    logger.info("🎬 开始实时推理循环...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠️ 无法读取摄像头帧")
                time.sleep(0.1)
                continue
            
            loop_start = time.time()
            
            # 1. 预处理
            preprocess_start = time.time()
            input_data = preprocess_matched_resolution(frame, dtype=np.float16)
            preprocess_time = (time.time() - preprocess_start) * 1000
            
            # 2. Atlas推理
            inference_start = time.time()
            outputs = model.infer([input_data])
            inference_time = (time.time() - inference_start) * 1000
            
            # 3. 后处理
            postprocess_start = time.time()
            lane_mask = postprocess_matched_resolution(outputs[0], actual_w, actual_h)
            postprocess_time = (time.time() - postprocess_start) * 1000
            
            # 4. 透视变换和路径规划
            transform_start = time.time()
            
            # 使用与单文件推理完全相同的逻辑，应用边缘计算优化
            if edge_computing:
                if full_image_bird_eye:
                    # 边缘计算+完整图像：超低像素密度
                    adjusted_pixels_per_unit = 1  # 固定1像素/单位，减少400倍计算量
                    print(f"⚡ 边缘计算极致优化：像素密度 {pixels_per_unit} → {adjusted_pixels_per_unit} 像素/单位")
                else:
                    # 边缘计算+A4区域：低像素密度
                    adjusted_pixels_per_unit = 2  # 固定2像素/单位
                    print(f"⚡ 边缘计算优化：像素密度 {pixels_per_unit} → {adjusted_pixels_per_unit} 像素/单位")
            else:
                if full_image_bird_eye:
                    # 完整图像模式：极低像素密度（边缘计算友好）
                    adjusted_pixels_per_unit = max(1, pixels_per_unit // 20)  # 最低1像素/单位，减少400倍计算量
                    print(f"📱 边缘计算优化：像素密度 {pixels_per_unit} → {adjusted_pixels_per_unit} 像素/单位（减少{pixels_per_unit//adjusted_pixels_per_unit}倍计算量）")
                else:
                    # A4纸区域模式：中等优化
                    adjusted_pixels_per_unit = max(2, pixels_per_unit // 4)  # 最低2像素/单位
                    print(f"🔧 性能优化：像素密度 {pixels_per_unit} → {adjusted_pixels_per_unit} 像素/单位")
            
            # 使用已初始化的transformer对象，避免重复创建
            bird_eye_image, bird_eye_mask, view_params = transformer.transform_image_and_mask(
                frame, lane_mask, pixels_per_unit=adjusted_pixels_per_unit, margin_ratio=margin_ratio, full_image=full_image_bird_eye)
            
            # 路径规划 - 使用与单文件推理相同的参数
            control_map, path_data = create_control_map(
                bird_eye_mask, view_params, 
                add_grid=True, add_path=True,
                path_smooth_method='polynomial',
                path_degree=3,
                num_waypoints=20,
                min_road_width=10,
                edge_computing=True, 
                force_bottom_center=True
            )
            transform_time = (time.time() - transform_start) * 1000
            
            # 检查Web界面参数更新
            if enable_web and enable_control and controller:
                with web_data_lock:
                    if web_data.get('params_updated', False):
                        # 应用新参数到控制器
                        new_params = web_data['control_params']
                        controller.steering_gain = new_params['steering_gain']
                        controller.base_pwm = new_params['base_speed']
                        controller.preview_distance = new_params['preview_distance']
                        controller.curvature_damping = new_params['curvature_damping']
                        
                        # 应用EMA平滑参数（如果有）
                        if 'ema_alpha' in new_params:
                            controller.update_smoothing_params(
                                ema_alpha=new_params['ema_alpha'],
                                enable_smoothing=new_params.get('enable_smoothing', controller.enable_smoothing)
                            )
                        
                        web_data['params_updated'] = False  # 重置标志
                        print(f"🎛️ 控制参数已更新: 转向增益={controller.steering_gain}, "
                              f"基础PWM={controller.base_pwm}, 预瞄距离={controller.preview_distance}cm, "
                              f"阻尼系数={controller.curvature_damping}, EMA平滑={'启用' if controller.enable_smoothing else '禁用'}(α={controller.ema_alpha})")
            
            # 5. 控制计算
            control_time = 0
            control_result = None
            if enable_control and path_data is not None and controller is not None:
                # 检查是否需要重置EMA状态（例如从紧急停车恢复或刚开始行驶）
                # 优化版本：只检查lateral_error的EMA状态
                with web_data_lock:
                    if (web_data.get('car_driving', False) and 
                        not web_data.get('emergency_stop', False) and
                        hasattr(controller, 'ema_lateral_error') and 
                        controller.ema_lateral_error is None):
                        # 如果刚开始行驶且EMA状态未初始化，则需要准备接受新的控制
                        print("🔄 开始行驶，EMA平滑器准备就绪（优化版本：输入信号平滑）")
                    elif web_data.get('emergency_stop', False):
                        # 紧急停车状态，重置EMA状态以避免残留影响
                        if hasattr(controller, 'reset_ema_state'):
                            controller.reset_ema_state()
                            print("🛑 紧急停车状态，EMA状态已重置（优化版本）")
                
                control_start = time.time()
                control_result = controller.compute_wheel_pwm(path_data, view_params)
                control_time = (time.time() - control_start) * 1000
            
            # 6. 串口控制指令发送
            if enable_web and control_result:
                with web_data_lock:
                    # 检查是否启用串口控制
                    if (web_data.get('car_driving', False) and 
                        web_data.get('serial_connected', False) and 
                        not web_data.get('emergency_stop', False)):
                        
                        # 发送控制指令到串口
                        if car_controller is not None and car_controller.is_connected:
                            try:
                                # 将PWM值转换为串口控制器需要的轮速
                                # PWM范围通常是0-1000，转换为串口控制器的-1000到+1000范围
                                left_speed = int(control_result['pwm_left'])
                                right_speed = int(control_result['pwm_right'])
                                
                                # 限制速度范围
                                left_speed = max(-1000, min(1000, left_speed))
                                right_speed = max(-1000, min(1000, right_speed))
                                
                                # 发送控制指令
                                success = car_controller.set_wheel_speeds(left_speed, right_speed)
                                
                                if success:
                                    # 更新最后发送的控制指令
                                    web_data['last_control_command'] = {
                                        'left_speed': left_speed,
                                        'right_speed': right_speed,
                                        'timestamp': time.time()
                                    }
                                    # 每50帧输出一次串口控制信息
                                    if frame_count % 50 == 0:
                                        print(f"📡 串口控制指令发送: 左轮={left_speed}, 右轮={right_speed}")
                                else:
                                    print(f"⚠️ 串口控制指令发送失败")
                                    
                            except Exception as e:
                                print(f"❌ 串口控制错误: {e}")
                                # 串口错误时自动停止行驶
                                web_data['car_driving'] = False
                                web_data['emergency_stop'] = True
                    
                    elif web_data.get('emergency_stop', False):
                        # 紧急停车状态下确保发送停止指令
                        if car_controller is not None and car_controller.is_connected:
                            try:
                                car_controller.stop()
                                web_data['last_control_command'] = {
                                    'left_speed': 0,
                                    'right_speed': 0,
                                    'timestamp': time.time()
                                }
                            except Exception as e:
                                print(f"❌ 紧急停车串口指令错误: {e}")
            
            # 性能统计
            frame_count += 1
            total_times["preprocess"] += int(preprocess_time)
            total_times["inference"] += int(inference_time)
            total_times["postprocess"] += int(postprocess_time)
            total_times["transform"] += int(transform_time)
            total_times["control"] += int(control_time)
            
            pipeline_latency = (time.time() - loop_start) * 1000
            
            # 每20帧输出一次详细统计
            if frame_count % 20 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed
                
                avg_preprocess = total_times["preprocess"] / frame_count
                avg_inference = total_times["inference"] / frame_count
                avg_postprocess = total_times["postprocess"] / frame_count
                avg_transform = total_times["transform"] / frame_count
                avg_control = total_times["control"] / frame_count
                avg_total = sum(total_times.values()) / frame_count
                
                logger.info(f"📊 第{frame_count}帧性能分析:")
                logger.info(f"   预处理: {preprocess_time:.1f}ms (平均: {avg_preprocess:.1f}ms)")
                logger.info(f"   Atlas推理: {inference_time:.1f}ms (平均: {avg_inference:.1f}ms)")
                logger.info(f"   后处理: {postprocess_time:.1f}ms (平均: {avg_postprocess:.1f}ms)")
                logger.info(f"   透视变换: {transform_time:.1f}ms (平均: {avg_transform:.1f}ms)")
                if enable_control:
                    logger.info(f"   控制计算: {control_time:.1f}ms (平均: {avg_control:.1f}ms)")
                logger.info(f"   总延迟: {pipeline_latency:.1f}ms (平均: {avg_total:.1f}ms)")
                logger.info(f"   实际FPS: {avg_fps:.1f}, 理论FPS: {1000/avg_total:.1f}")
                
                # 控制信息
                if control_result:
                    logger.info(f"🚗 控制指令: 左轮={control_result['pwm_left']:.0f}, 右轮={control_result['pwm_right']:.0f}")
                    logger.info(f"   横向误差: {control_result['lateral_error']:.2f}cm, 曲率: {control_result.get('curvature_level', 0):.4f}")
            
            # 检测车道线
            lane_pixels = np.sum(lane_mask > 0)
            total_pixels = lane_mask.shape[0] * lane_mask.shape[1]
            lane_ratio = (lane_pixels / total_pixels) * 100
            
            # 每帧简要日志
            if control_result:
                logger.info(f"帧{frame_count}: 延迟{pipeline_latency:.1f}ms, 车道线{lane_ratio:.1f}%, "
                          f"控制[L:{control_result['pwm_left']:.0f}, R:{control_result['pwm_right']:.0f}]")
                # 每帧详细控制信息
                logger.info(f"   🚗 横向误差: {control_result['lateral_error']:.2f}cm, "
                          f"曲率: {control_result.get('curvature_level', 0):.4f}, "
                          f"转向: {control_result.get('turn_direction', 'unknown')}")
            else:
                logger.info(f"帧{frame_count}: 延迟{pipeline_latency:.1f}ms, 车道线{lane_ratio:.1f}%")
            
            # 更新Web界面数据
            if enable_web:
                with web_data_lock:
                    web_data['frame_count'] = frame_count
                    
                    # 调试信息
                    if control_map is not None:
                        print(f"🖼️ 生成控制地图: {control_map.shape}, 数据类型: {control_map.dtype}")
                        web_data['latest_control_map'] = control_map.copy()
                    else:
                        print("⚠️ 控制地图为None")
                        web_data['latest_control_map'] = None
                        
                    web_data['latest_stats'] = {
                        'latency': pipeline_latency,
                        'lane_ratio': lane_ratio,
                        'left_pwm': control_result['pwm_left'] if control_result else 0,
                        'right_pwm': control_result['pwm_right'] if control_result else 0,
                        'lateral_error': control_result['lateral_error'] if control_result else 0,
                        'path_curvature': control_result.get('curvature_level', 0) if control_result else 0,
                        # 串口控制状态
                        'serial_connected': web_data.get('serial_connected', False),
                        'car_driving': web_data.get('car_driving', False),
                        'control_enabled': web_data.get('control_enabled', False),
                        'last_command_sent': (web_data.get('last_control_command') or {}).get('timestamp', 0)
                    }
            
            # 检测退出条件（仅在有GUI时检查按键）
            if not no_gui:
                try:
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC键
                        logger.info("🛑 用户按ESC键退出")
                        break
                except cv2.error:
                    # 如果OpenCV GUI不可用，忽略错误
                    logger.warning("⚠️ OpenCV GUI不可用，无法检测按键")
                    no_gui = True  # 自动切换到无GUI模式
            else:
                # 无GUI模式下可以通过其他方式退出，例如文件标志
                time.sleep(0.001)  # 短暂休眠避免过度占用CPU
                
    except KeyboardInterrupt:
        logger.info("🛑 用户中断 (Ctrl+C)")
    except Exception as e:
        logger.error(f"❌ 实时推理错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # 更新Web界面状态
        if enable_web:
            with web_data_lock:
                web_data['is_running'] = False
                web_data['car_driving'] = False
                web_data['emergency_stop'] = True
        
        # 安全关闭车辆控制器
        if car_controller is not None:
            try:
                car_controller.stop()  # 发送停止指令
                car_controller.disconnect()  # 断开串口连接
                logger.info("🔌 车辆控制器已安全关闭")
            except Exception as e:
                logger.error(f"⚠️ 关闭车辆控制器时出错: {e}")
                
        cap.release()
        
        # 仅在非无GUI模式下调用OpenCV GUI函数
        if not no_gui:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                # 忽略OpenCV GUI相关错误
                pass
                
        logger.info("🔚 实时推理系统已关闭") 