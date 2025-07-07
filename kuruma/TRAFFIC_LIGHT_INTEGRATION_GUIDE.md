# 交通灯检测集成指南

## 概述

本指南描述了如何将交通灯检测功能集成到kuruma控制系统中。该功能基于Atlas昇腾推理，实现了红绿灯检测并与车辆状态机集成，实现红灯停车、绿灯通行的智能控制。

## 功能特性

- 🚦 **智能交通灯检测**: 基于Atlas昇腾NPU的YOLOv5模型推理
- 🚗 **状态机集成**: 与现有避障状态机无缝集成
- ⏱️ **按需检测**: 每10帧执行一次检测，优化性能
- 🔄 **状态切换**: 红灯停车 ↔ 绿灯通行自动切换
- 📊 **实时监控**: Web界面实时显示交通灯状态
- 🛡️ **优先级管理**: 红灯检测优先级高于障碍物检测

## 核心逻辑

### 状态机扩展

原有状态机已扩展支持交通灯检测：

```
LANE_FOLLOWING (巡线模式)
    ↓ 检测到红灯
RED_LIGHT_WAITING (红灯等待)
    ↓ 检测到绿灯
LANE_FOLLOWING (返回巡线)
```

### 检测策略

1. **每10帧检测**: 默认每10帧执行一次交通灯检测
2. **红灯优先**: 在巡线模式下，红灯检测优先级最高
3. **绿灯唤醒**: 仅在红灯等待状态检查绿灯
4. **忽略绿灯**: 在正常巡线时忽略绿灯检测

## 文件修改清单

### 1. 状态机扩展 (`kuruma/control/state_machine.py`)

- 添加 `RED_LIGHT_WAITING` 状态
- 扩展 `process_frame` 方法支持交通灯检测参数
- 添加 `is_traffic_light_detection_frame` 方法
- 更新控制决策逻辑

### 2. 交通灯检测模块 (`kuruma/vision/traffic_light_detection.py`)

- 新建交通灯检测器类 `TrafficLightDetector`
- 集成Atlas昇腾推理逻辑
- 实现红绿灯状态判断
- 提供工厂函数 `create_traffic_light_detector`

### 3. 实时推理接口 (`kuruma/interfaces/realtime.py`)

- 添加交通灯检测初始化
- 集成检测循环逻辑
- 更新状态机调用
- 添加性能统计
- 更新Web界面数据

### 4. 主控制脚本 (`kuruma/kuruma_control_dashboard.py`)

- 添加命令行参数支持
- 传递交通灯检测配置
- 支持启用/禁用功能

## 使用方法

### 1. 基本使用

```bash
# 启用交通灯检测的实时模式（默认启用）
python kuruma_control_dashboard.py --realtime

# 禁用交通灯检测
python kuruma_control_dashboard.py --realtime --disable_traffic_light_detection

# 自定义检测间隔（每15帧检测一次）
python kuruma_control_dashboard.py --realtime --traffic_light_detection_interval 15
```

### 2. Web界面监控

启用Web界面查看交通灯状态：

```bash
python kuruma_control_dashboard.py --realtime --web --web_port 5000
```

访问 `http://localhost:5000` 查看：
- 交通灯检测状态
- 当前状态（红灯/绿灯/未知）
- 检测置信度
- 状态机当前模式

### 3. 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable_traffic_light_detection` | True | 启用交通灯检测 |
| `--disable_traffic_light_detection` | False | 禁用交通灯检测 |
| `--traffic_light_detection_interval` | 10 | 检测间隔帧数 |

## 配置说明

### 1. 模型路径

交通灯检测器自动使用以下路径：
- 模型: `atlasyolo/best.om`
- 标签: `atlasyolo/labels.txt`

### 2. 检测阈值

默认检测参数：
- 置信度阈值: 0.4
- IoU阈值: 0.5
- 红绿灯判断阈值: 0.5

### 3. 状态切换逻辑

```python
# 红灯检测（巡线模式下）
if traffic_light_status == "red" and confidence > 0.5:
    state = RED_LIGHT_WAITING

# 绿灯检测（红灯等待模式下）
if traffic_light_status == "green" and confidence > 0.5:
    state = LANE_FOLLOWING
```

## 性能优化

### 1. 检测间隔

- 默认每10帧检测一次，减少计算负担
- 可根据实际需求调整间隔
- 建议范围：5-20帧

### 2. 推理时间

典型性能指标：
- 交通灯检测: ~20-30ms
- 总延迟增加: <5%
- 内存占用增加: 最小

### 3. 错误处理

- 检测失败时自动降级
- 模型加载失败时禁用功能
- 实时错误日志记录

## 日志输出示例

```
🚦 交通灯检测器初始化成功
   模型: /path/to/best.om
   标签: {0: 'car', 1: 'red', 2: 'green'}
   设备: 0

🚦 第100帧检测到交通灯: RED, 置信度0.85
🔄 状态切换: lane_following → red_light_waiting
🤖 状态机控制: 红灯等待中，停车等待

🚦 第200帧检测到交通灯: GREEN, 置信度0.92
🔄 状态切换: red_light_waiting → lane_following
🚗 状态机控制: 正常巡线模式
```

## 故障排除

### 1. 常见问题

**Q: 交通灯检测器初始化失败**
```
❌ 创建交通灯检测器失败: 模型文件不存在
```
A: 确保 `atlasyolo/best.om` 和 `atlasyolo/labels.txt` 文件存在

**Q: 检测不准确**
```
🔍 第50帧未检测到交通灯
```
A: 检查光照条件、相机角度或调整检测阈值

**Q: 状态切换异常**
```
🛡️ 状态机保护机制激活
```
A: 检查状态机日志，可能触发了防死循环保护

### 2. 调试模式

启用详细日志：
```bash
python kuruma_control_dashboard.py --realtime --log_level DEBUG
```

### 3. 测试工具

使用独立测试脚本：
```bash
cd kuruma
python test_traffic_light_atlas.py
```

## 扩展功能

### 1. 添加黄灯检测

1. 更新 `atlasyolo/labels.txt` 添加 `yellow` 类别
2. 在 `TrafficLightDetector` 中添加黄灯判断逻辑
3. 扩展状态机添加 `YELLOW_LIGHT_CAUTION` 状态

### 2. 集成GPS定位

结合GPS数据，在特定路口启用交通灯检测：
```python
if gps_data.near_intersection():
    enable_traffic_light_detection = True
```

### 3. 多模态融合

结合其他传感器数据提高检测准确性：
- 激光雷达停车线检测
- 声音识别（交通灯提示音）
- V2X通信（车路协同）

## 最佳实践

### 1. 部署建议

- 在良好光照条件下测试
- 确保相机视野包含交通灯区域
- 定期校准相机参数
- 监控检测准确率

### 2. 安全考虑

- 保持人工监控能力
- 设置紧急停车机制
- 定期检查系统状态
- 建立故障恢复流程

### 3. 维护指南

- 定期更新模型权重
- 清理日志文件
- 监控系统性能

## 最新更新 (v3.0) - 多模型架构

### 重大升级：单进程多线程多Stream架构

#### 1. 统一Atlas会话管理器

创建了 `kuruma/core/atlas_session_manager.py` 来解决多模型冲突问题：

**核心特性**:
- **单例模式**: 全局唯一的Atlas环境管理
- **线程安全**: 支持多线程并发模型加载
- **模型缓存**: 避免重复加载相同模型
- **独立Stream**: 每个模型使用独立的推理流

**解决的问题**:
- ✅ 修复 `aclInit` 重复调用冲突
- ✅ 解决 `'NoneType' object has no attribute 'infer'` 错误
- ✅ 实现真正的多模型并发推理
- ✅ 统一资源管理和清理

#### 2. 多模型并发架构

```
                    Atlas NPU设备
                         |
                Atlas会话管理器 (单例)
                    /           \
            车道线分割模型      交通灯检测模型
               Stream A          Stream B
                 |                   |
            主推理循环          每10帧检测
```

**技术实现**:
- **主线程**: 初始化Atlas环境，加载所有模型
- **推理线程**: 每个模型在独立线程中运行
- **Stream隔离**: 使用不同的Stream避免推理冲突
- **内存管理**: 每个模型独立的输入输出Tensor

#### 3. Web界面交通灯状态显示

新增专门的交通灯状态面板，实时显示：
- **交通灯检测状态**: 启用/禁用
- **检测结果**: 是否检测到交通灯
- **交通灯状态**: red/green/unknown（带颜色区分）
- **检测置信度**: 精确到小数点后2位
- **下次检测倒计时**: 距离下次检测的帧数

#### 4. 工具函数本地化

创建了 `kuruma/vision/traffic_light_utils.py`，包含：
- `letterbox`: 图像预处理
- `scale_coords`: 坐标缩放
- `nms`: 非极大值抑制
- `get_labels_from_txt`: 标签文件读取
- 其他YOLOv5兼容工具函数

**优势**:
- 不再依赖atlasyolo目录导入
- 避免路径问题
- 更好的模块化

#### 5. 智能模型路径检测

支持多个模型文件位置，按优先级自动检测：
1. `car/best.om` 和 `car/labels.txt` (推荐)
2. `atlasyolo/best.om` 和 `atlasyolo/labels.txt` (备选)

#### 6. 增强的Web界面样式

新增交通灯专用CSS样式：
- 红灯状态：红色背景 (#FF4444)
- 绿灯状态：绿色背景 (#4CAF50)
- 未知状态：橙色背景 (#FFA500)

### 使用示例

#### Web界面监控

```bash
# 启动带Web界面的交通灯检测
python kuruma_control_dashboard.py --realtime --web --web_port 5000
```

访问 `http://localhost:5000` 查看交通灯状态面板

### 测试新架构

#### 1. 多模型架构测试

运行专门的测试脚本验证多模型并发：

```bash
# 测试多模型Atlas推理
python kuruma/test_multi_model_atlas.py
```

**测试项目**:
- Atlas会话管理器初始化
- 车道线分割模型加载和推理
- 交通灯检测模型加载和推理
- 并发推理性能测试

#### 2. 生产环境运行

```bash
# 完整功能启动（推荐）
python kuruma_control_dashboard.py --realtime --web --no_gui \
  --enable_control --enable_obstacle_detection \
  --enable_traffic_light_detection \
  --log_file realtime_control.log \
  --enable_serial --auto_connect_serial \
  --edge_computing --enable_smoothing --ema_alpha 0.5
```

#### 3. 故障排除

**常见问题**:
1. **模型加载失败**: 检查模型文件路径和Atlas环境
2. **推理冲突**: 确保使用新的会话管理器
3. **内存不足**: 监控NPU内存使用情况
4. **线程死锁**: 检查并发访问的资源锁

**调试命令**:
```bash
# 检查Atlas设备状态
npu-smi info

# 查看详细日志
tail -f realtime_control.log

# 测试模型文件
python -c "from kuruma.core.atlas_session_manager import get_atlas_session; print(get_atlas_session().list_models())"
```

#### 模型文件配置

```bash
# 推荐：将模型文件放在car目录
cp best.om car/
cp labels.txt car/

# 或者使用atlasyolo目录
cp best.om atlasyolo/
cp labels.txt atlasyolo/
```

### 技术改进

#### 1. 错误处理增强

- 优雅的模型加载失败处理
- 工具函数导入失败的备用方案
- 详细的错误信息提示

#### 2. 性能优化

- 交通灯检测倒计时避免不必要的计算
- Web界面异步更新减少阻塞
- 内存使用优化

#### 3. 日志改进

更详细的日志输出：
```
✅ 交通灯检测工具函数导入成功
🚦 交通灯检测器初始化成功
   模型: /path/to/car/best.om
   标签: {0: 'vehicle', 1: 'red', 2: 'green'}
   设备: 0
🚦 第100帧交通灯检测: red(0.85), 下次检测: 10帧后
```

### 兼容性说明

- 完全向后兼容现有配置
- 自动检测模型文件位置
- 保持原有命令行参数接口
- Web界面向下兼容旧版浏览器

### 升级建议

1. **模型文件迁移**: 建议将模型文件移动到 `car/` 目录
2. **工具函数更新**: 系统自动使用新的本地工具函数
3. **Web界面体验**: 启用Web界面查看完整的交通灯状态
4. **日志监控**: 关注新的日志格式和状态信息
- 备份配置文件

## 技术支持

如有问题或建议：
1. 查看系统日志获取详细错误信息
2. 确认Atlas昇腾环境配置
3. 验证模型文件完整性
4. 检查相机和网络连接

---

**版本**: v1.0  
**最后更新**: 2025年1月  
**兼容性**: Atlas昇腾NPU, Python 3.9+ 