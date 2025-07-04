# 感知与环境建模完成报告

## 🎯 项目目标
将图像转换成供路径规划使用的、上帝视角的2D地图

## ✅ 已完成功能

### 1. 🧠 车道线分割推理
- **模型**: Fast-SCNN ONNX模型 (640×360)
- **功能**: 检测可驾驶区域
- **性能**: ~9-10 FPS (CPU), 理论支持GPU加速
- **输出**: 分割掩码 + 可视化结果

### 2. 🦅 透视变换 (逆透视变换 IPM)
- **标定方式**: 基于A4纸标定点
- **标定参数**: 
  - 图像点: [(260, 87), (378, 87), (410, 217), (231, 221)]
  - 世界点: [(0, 0), (21, 0), (21, 29.7), (0, 29.7)] cm
- **输出尺寸**: 自动计算 (例: 776×950像素)
- **世界坐标范围**: X(-8.9~29.9), Y(-8.9~38.6) cm

### 3. 🗺️ 控制地图生成
- **网格系统**: 10cm间隔网格
- **坐标标注**: 完整的X-Y坐标系
- **标记点**: A4纸四个角点标记
- **可驾驶区域**: 绿色标识

## 📁 输出文件说明

**所有输出文件统一保存在 `output/` 目录中**

### 基础分割结果
- `output/{filename}_onnx_result.jpg` - 原图 + 可驾驶区域覆盖
- `output/{filename}_onnx_mask.jpg` - 纯分割掩码 (可选)

### 鸟瞰图结果
- `output/{filename}_bird_eye.jpg` - 原图的鸟瞰图
- `output/{filename}_bird_eye_segmented.jpg` - 鸟瞰图 + 可驾驶区域分割
- `output/{filename}_control_map.jpg` - 控制地图 (带网格坐标系)

## 🚀 使用方法

### 基础车道线分割
```bash
python onnx_single_image_inference.py --input image.jpg --output result.jpg
```

### 完整感知管道 (推荐)
```bash
python onnx_single_image_inference.py \
    --input image.jpg \
    --output result.jpg \
    --bird_eye \
    --save_control_map
```

### 参数说明
- `--bird_eye`: 启用透视变换
- `--save_control_map`: 生成控制地图
- `--pixels_per_unit 20`: 每厘米像素数 (默认20)
- `--margin_ratio 0.3`: 边距比例 (默认0.3)
- `--preview`: 显示预览窗口

## 📊 性能数据 (测试图像: 640×360)

```
⏱️  图片加载           :    1.0ms (  0.6%)
⏱️  模型加载           :   54.0ms ( 35.0%)  
⏱️  CPU预处理         :   16.3ms ( 10.6%)
⏱️  ONNX推理         :   34.0ms ( 22.0%)
⏱️  CPU后处理         :    5.0ms (  3.2%)
⏱️  透视变换           :   26.0ms ( 16.8%)
⏱️  结果保存           :   18.1ms ( 11.7%)
--------------------------------------------
🏁 总耗时: 154.4ms
⚡ 理论FPS: 6.5
```

## 🔧 技术细节

### 透视变换矩阵
- **计算方法**: OpenCV `cv2.getPerspectiveTransform()`
- **内置标定**: 基于A4纸的4点标定
- **自动缩放**: 支持不同输入图像尺寸
- **坐标系**: 以A4纸左上角为原点

### 控制地图特性
- **分辨率**: 可配置 (默认20像素/cm)
- **网格**: 10cm间隔
- **颜色编码**: 
  - 绿色: 可驾驶区域
  - 黑色: 不可驾驶区域
  - 灰色: 网格线
  - 红色: 原点标记
  - 黄色: 标定点

## 🎯 后续应用

这个2D地图可以直接用于：
1. **路径规划**: A*算法、RRT等
2. **避障控制**: 基于占栅格地图
3. **车辆定位**: 相对坐标定位
4. **决策控制**: 转向和速度控制

## 📝 测试结果示例

测试图像: `raw_20250702_024221_396.jpg`
- **检测覆盖率**: 50.94% (原图) / 48.71% (鸟瞰图)
- **鸟瞰图尺寸**: 776×950像素
- **世界坐标范围**: 38.8cm × 47.5cm

## ✅ 总结

**第一部分：感知与环境建模已完成！**

我们成功实现了从原始相机图像到2D控制地图的完整管道：
1. ✅ 车道线/可驾驶区域检测
2. ✅ 透视变换标定与应用
3. ✅ 鸟瞰图生成
4. ✅ 控制地图生成
5. ✅ 完整的坐标系和网格

现在可以进入下一阶段：**路径规划与车辆控制**！
