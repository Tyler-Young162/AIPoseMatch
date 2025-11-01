# 安装指南 / Installation Guide

## Windows系统安装步骤

### 第一步：安装Python

1. 访问 Python官网: https://www.python.org/downloads/
2. 下载 Python 3.9 或更高版本
3. 运行安装程序，**务必勾选 "Add Python to PATH"**
4. 验证安装：
   ```bash
   python --version
   ```

### 第二步：安装CUDA和PyTorch

#### 2.1 安装NVIDIA显卡驱动

1. 访问 NVIDIA官网: https://www.nvidia.com/drivers
2. 下载并安装最新的显卡驱动
3. 确保您的显卡支持CUDA

#### 2.2 安装CUDA Toolkit

1. 访问 NVIDIA CUDA: https://developer.nvidia.com/cuda-downloads
2. 选择Windows版本
3. 下载并安装CUDA Toolkit 11.8或12.1
4. 验证安装：
   ```bash
   nvcc --version
   ```

#### 2.3 安装PyTorch

1. 访问 PyTorch官网: https://pytorch.org/get-started/locally/
2. 选择配置：
   - Package: **Pip**
   - OS: **Windows**
   - Compute Platform: **CUDA 11.8** 或 **CUDA 12.1**
3. 复制安装命令并执行：

**CUDA 11.8示例:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1示例:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. 验证PyTorch和CUDA：
   ```bash
   python
   ```
   ```python
   import torch
   print(torch.cuda.is_available())  # 应该输出 True
   print(torch.cuda.get_device_name(0))  # 显示GPU名称
   ```

### 第三步：设置项目

1. 打开命令提示符或PowerShell
2. 进入项目目录：
   ```bash
   cd C:\Users\yytsk\Documents\AIPoseMatch
   ```

3. 创建虚拟环境：
   ```bash
   python -m venv venv
   ```

4. 激活虚拟环境：
   ```bash
   venv\Scripts\activate
   ```

### 第四步：安装依赖

1. 安装所有依赖包：
   ```bash
   pip install -r requirements.txt
   ```

2. 验证依赖安装：
   ```bash
   python test_dependencies.py
   ```

如果所有依赖都安装成功，您会看到：
```
✓ PyTorch installed
✓ TorchVision installed
✓ OpenCV installed
✓ MediaPipe installed
...
All dependencies are installed!
✓ CUDA is available
```

### 第五步：配置摄像头

1. 编辑 `config.yaml` 文件
2. 找到 `camera` 部分：
   ```yaml
   camera:
     device_id: 0  # 通常是0，如果有多个摄像头可尝试1,2等
     resolution: [1280, 720]
     fps: 30
   ```

3. 如果摄像头无法打开，尝试修改 `device_id`

### 第六步：运行程序

```bash
python src/main.py
```

### 常见问题解决

#### Q1: 导入错误 `ImportError: No module named 'cv2'`

**A**: 重新安装OpenCV：
```bash
pip install opencv-python --upgrade
```

#### Q2: 导入错误 `ImportError: No module named 'mediapipe'`

**A**: 重新安装MediaPipe：
```bash
pip install mediapipe --upgrade
```

#### Q3: CUDA不可用 `torch.cuda.is_available()` 返回False

**A**: 检查以下问题：
1. 是否正确安装了CUDA Toolkit
2. PyTorch安装时选择的CUDA版本是否与已安装的版本匹配
3. 显卡驱动是否最新
4. 显卡是否支持CUDA

#### Q4: 摄像头无法打开

**A**: 尝试以下方法：
1. 确认摄像头没有被其他程序占用（关闭QQ、微信、浏览器等）
2. 在config.yaml中尝试不同的device_id（0, 1, 2...）
3. 在Windows设备管理器中确认摄像头正常工作

#### Q5: 程序运行很慢（帧率低）

**A**: 可以尝试：
1. 降低摄像头分辨率：`resolution: [640, 480]`
2. 缩小ROI区域
3. 降低matting质量：`downsample_ratio: 0.5`
4. 如果使用CPU，安装带CUDA的版本以获得GPU加速

#### Q6: MediaPipe检测不到人体

**A**: 调整配置：
1. 增加光照
2. 降低检测阈值：`min_detection_confidence: 0.3`
3. 确保人体在ROI范围内
4. 检查摄像头是否对焦正常

### 性能优化建议

1. **GPU加速**: 确保CUDA正常工作以获得最佳性能
2. **分辨率**: 降低分辨率可提高帧率
3. **ROI大小**: 减小ROI区域减少处理量
4. **模型复杂度**: MediaPipe默认使用model_complexity=1，可以改为0提高速度

### 技术支持

如果遇到问题：
1. 先运行 `python test_dependencies.py` 检查依赖
2. 查看QUICKSTART.md了解更多使用方法
3. 检查config.yaml配置是否正确
4. 查看README.md了解项目概述

