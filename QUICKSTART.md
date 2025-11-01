# AI Pose Match - Quick Start Guide

## 快速开始指南

### 环境要求
- Windows 10/11
- Python 3.9 或更高版本
- NVIDIA GPU（支持CUDA）
- 网络摄像头

### 安装步骤

#### 1. 安装CUDA和PyTorch

首先安装适合您系统的PyTorch版本：

**访问 PyTorch 官网**: https://pytorch.org/get-started/locally/

选择您的配置：
- Package: Pip
- OS: Windows
- Compute Platform: CUDA 11.8 或 CUDA 12.1

例如，安装PyTorch with CUDA 11.8：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. 创建虚拟环境

```bash
python -m venv venv
venv\Scripts\activate
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### 4. 配置摄像头

编辑 `config.yaml` 文件：

```yaml
camera:
  device_id: 0  # 改为您的摄像头ID（通常0是默认摄像头）
  resolution: [1280, 720]
  fps: 30
```

#### 5. 运行程序

```bash
python src/main.py
```

### 功能使用

#### ROI设置
在 `config.yaml` 中调整有效区域：

```yaml
roi:
  x_min: 0.4  # 左边距（0.0-1.0）
  x_max: 0.6  # 右边距（0.0-1.0）
  y_min: 0.0  # 上边距（0.0-1.0）
  y_max: 0.8  # 下边距（0.0-1.0）
```

#### 键盘控制

- **Q**: 退出程序
- **S**: 保存当前帧到文件
- **R**: 切换ROI显示

### 显示说明

程序将显示4个窗口面板：

1. **Original**: 原始画面，带ROI标记
2. **ROI**: 提取的有效区域
3. **Pose**: 人体骨骼检测结果
4. **Matting**: 人体抠像结果

左上角显示：
- **FPS**: 当前帧率
- **Detections**: 检测到的人数
- **Score**: 选择评分（完整性、高度、居中度）
- **Keypoints**: 可见关键点数量

### 高级配置

#### 人物选择权重

调整 `config.yaml` 中的人物选择权重：

```yaml
person_selection:
  completeness_weight: 0.4  # 完整性权重
  height_weight: 0.3        # 高度权重
  centeredness_weight: 0.3  # 居中度权重
  min_keypoints_visible: 8  # 最少可见关键点数
```

#### 性能优化

如果帧率低于预期，可以调整：

```yaml
matting:
  downsample_ratio: 0.5  # 降低分辨率以提高速度（0.25-1.0）
  
pose:
  model_complexity: 0  # 降低模型复杂度（0=快，2=准确）
```

### 故障排除

#### 摄像头无法打开

1. 检查 `device_id` 是否正确
2. 确认摄像头没有被其他程序占用
3. 尝试不同的 `device_id` 值（0, 1, 2...）

#### CUDA错误

1. 确认安装了NVIDIA显卡驱动
2. 验证CUDA工具包已安装
3. 检查PyTorch是否正确识别GPU：`python -c "import torch; print(torch.cuda.is_available())"`

#### 检测不到人体

1. 调整光照条件
2. 调整 `min_detection_confidence` 值
3. 确保人体在ROI范围内

#### 性能问题

1. 降低摄像头分辨率
2. 使用较小的ROI
3. 降低matting质量
4. 关闭不必要的显示面板

### 注意事项

- **当前版本使用简化的抠像算法**（基于HSV颜色空间）
- 对于生产环境，建议集成完整的RVM或其他专业抠像模型
- 请根据实际需求调整配置参数

### 下一步

如需更高级的功能或优化，可以考虑：

1. 集成完整的RVM模型用于更准确的抠像
2. 添加人体跟踪以增强稳定性
3. 实现背景替换功能
4. 添加骨骼动作识别

祝使用愉快！

