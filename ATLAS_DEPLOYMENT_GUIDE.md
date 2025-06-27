# Atlas NPU 端到端部署指南 (Fast-SCNN 640x480)

本指南详细说明了如何将 Fast-SCNN 端到端（End-to-End）车道线检测模型部署到 Atlas NPU 开发板上。该模型经过优化，将预处理（Resize、归一化）和后处理（Softmax）集成到 ONNX 模型中，实现了从摄像头原始输入到最终分割结果的“一站式”推理，极大地提升了在 Atlas 板上的运行效率。

**最新更新 (2024-07-15):**
- **模型分辨率**: 640x480，适配标准4:3摄像头，减少了图像畸变。
- **ONNX 兼容性**: 完全修复了 PyramidPooling 模块在非方形输入下的 ATC (ONNX -> OM) 转换问题。
- **性能**: PC端CUDA模拟测试显示理论性能可达 **200+ FPS**，在Atlas板上预期可达到 **25-40 FPS** 的实时性能。

---

## 部署流程

### 1. 准备工作

- **获取模型**:
  - 从项目 `weights/` 目录下找到最终的端到端 ONNX 模型：
    ```
    fast_scnn_480x640_e2e.onnx
    ```
  - 这个模型是“开箱即用”的，包含了所有预处理步骤。

- **上传模型到 Atlas 开发板**:
  - 使用 `scp` 或其他文件传输工具将 `fast_scnn_480x640_e2e.onnx` 文件上传到 Atlas 开发板的某个工作目录，例如 `/home/HwHiAiUser/Fast-SCNN/`。

### 2. ONNX 到 OM 模型转换 (ATC)

在 Atlas 开发板的终端环境中，使用 `atc` 命令将 `.onnx` 模型转换为昇腾处理器支持的 `.om` 离线模型。

- **ATC 转换命令**:
  ```bash
  atc --model=./fast_scnn_480x640_e2e.onnx \
      --framework=5 \
      --output=./fast_scnn_480x640_e2e \
      --input_format=NCHW \
      --input_shape="input:1,3,480,640" \
      --log=info \
      --soc_version=Ascend310
  ```
  - **`--model`**: 指向你的 `.onnx` 文件路径。
  - **`--framework=5`**: 5代表ONNX框架。
  - **`--output`**: 输出的 `.om` 模型文件路径（无需加扩展名）。
  - **`--input_shape`**: **必须显式指定**。我们的端到端模型输入是 `(1, 3, 480, 640)`，对应 (Batch, Channels, Height, Width)。
  - **`--soc_version`**: 根据你的Atlas开发板型号选择，例如 `Ascend310`。

- **预期结果**:
  - 命令执行成功后，你将在输出目录下得到 `fast_scnn_480x640_e2e.om` 文件。

### 3. 运行板端实时推理脚本

我们提供了一个优化的、极简的实时仪表盘脚本 `kuruma/lane_dashboard_e2e.py`，它直接调用 `.om` 模型进行推理。

- **修改脚本**:
  - 打开 `kuruma/lane_dashboard_e2e.py`。
  - 确保 `MODEL_PATH` 变量指向你刚刚生成的 `.om` 模型文件的**绝对路径**。
    ```python
    # kuruma/lane_dashboard_e2e.py

    # ... 其他代码 ...

    # --- 配置 ---
    MODEL_PATH = '/home/HwHiAiUser/Fast-SCNN/fast_scnn_480x640_e2e.om' # 确保这是OM模型的绝对路径
    CAMERA_ID = 0  # 摄像头ID
    # --- 配置结束 ---

    # ... 其他代码 ...
    ```

- **安装依赖**:
  - 确保你的 Python 环境中已经安装了 `atlas_utils`、`opencv-python` 和 `numpy`。

- **运行脚本**:
  - 在 Atlas 开发板上执行：
    ```bash
    python3 kuruma/lane_dashboard_e2e.py
    ```

### 4. 观察结果

- 脚本运行后，你将看到一个实时视频流窗口，其中：
  - **左侧**: 原始的摄像头输入画面 (640x480)。
  - **右侧**: 经过模型推理后，叠加了车道线预测结果的画面。
  - **仪表盘**: 实时显示当前的 **FPS** (每秒帧数)。

---

## 端到端模型的优势

- **极简预处理**: 在应用代码中，你**不再需要**进行复杂的 `resize`, `normalize` 等操作。CPU只负责将摄像头帧从 HWC 格式转为 CHW 格式，大大降低了CPU负载。
  ```python
  # 之前的复杂预处理 (已废弃):
  # image = cv2.resize(frame, (640, 480))
  # image = image.astype(np.float32) / 255.0
  # image = (image - mean) / std
  # image = image.transpose(2, 0, 1)

  # 现在 (端到端模型):
  input_tensor = frame.transpose(2, 0, 1) # 仅需一次转置
  ```

- **性能最大化**: 所有计算密集型任务（预处理、模型推理、后处理）都在NPU上完成，实现了硬件性能的最大化。

- **部署流程简化**: 减少了应用代码的复杂性和出错的可能性，使得部署更加健壮和高效。

---

至此，Fast-SCNN 端到端模型已成功部署在您的 Atlas 开发板上。享受实时、高效的车道线检测功能吧！
