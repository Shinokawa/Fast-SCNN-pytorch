# 模型和标签文件设置指南

## 文件放置位置

### 1. 推荐位置（优先级从高到低）

1. **car目录**（当前推荐）
   - 模型文件：`car/best.om`
   - 标签文件：`car/labels.txt`

2. **atlasyolo目录**（备选）
   - 模型文件：`atlasyolo/best.om`
   - 标签文件：`atlasyolo/labels.txt`

### 2. 自动检测逻辑

系统会自动按以下顺序查找模型文件：
1. 首先检查 `car/best.om` 和 `car/labels.txt`
2. 如果不存在，检查 `atlasyolo/best.om` 和 `atlasyolo/labels.txt`
3. 如果都不存在，会报错提示文件不存在

## 标签文件格式

### labels.txt 示例
```
vehicle
red
green
person
...
```

- 每行一个类别名称
- 类别ID为行号（从0开始）
- 例如：red对应类别ID 1，green对应类别ID 2

## 模型文件要求

- 格式：Atlas昇腾 `.om` 格式
- 输入尺寸：640x640（推荐）
- 输出格式：YOLOv5兼容格式

## 工具函数集成

交通灯检测模块现在使用本地工具函数，不再依赖atlasyolo目录的导入：
- 工具函数位置：`kuruma/vision/traffic_light_utils.py`
- 包含：letterbox、scale_coords、nms等必要函数

## 使用方法

### 1. 手动指定路径
```bash
# 创建交通灯检测器时指定路径
detector = create_traffic_light_detector(
    model_path='/path/to/best.om',
    labels_path='/path/to/labels.txt'
)
```

### 2. 使用默认自动检测
```bash
# 系统会自动查找模型和标签文件
detector = create_traffic_light_detector()
```

### 3. 在主程序中启用交通灯检测
```bash
# 启用交通灯检测（默认启用）
python kuruma_control_dashboard.py --realtime --enable_traffic_light_detection

# 禁用交通灯检测
python kuruma_control_dashboard.py --realtime --disable_traffic_light_detection
```

## 状态显示

交通灯检测状态会在以下地方显示：
1. **命令行日志**：检测结果和推理时间
2. **Web界面**：实时状态面板显示
   - 交通灯检测：启用/禁用
   - 检测到交通灯：有/无
   - 交通灯状态：red/green/unknown
   - 检测置信度：0.0-1.0
   - 下次检测倒计时：帧数

## 故障排除

### 1. 模型文件不存在
```
FileNotFoundError: 模型文件不存在: xxx/best.om
```
**解决方法**：将模型文件复制到推荐位置或指定正确路径

### 2. 标签文件不存在
```
FileNotFoundError: 标签文件不存在: xxx/labels.txt
```
**解决方法**：创建标签文件或指定正确路径

### 3. 工具函数导入失败
```
ImportError: 无法导入交通灯检测工具函数
```
**解决方法**：确保 `kuruma/vision/traffic_light_utils.py` 文件存在

### 4. Atlas环境问题
```
ImportError: Atlas昇腾环境不可用
```
**解决方法**：安装mindx.sdk库或检查Atlas环境配置

## 配置建议

1. **模型文件**：建议放在 `car/` 目录下，便于管理
2. **权限设置**：确保文件有读取权限
3. **路径检查**：定期检查文件是否存在
4. **备份**：对重要模型文件进行备份 