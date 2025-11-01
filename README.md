# AI Pose Match - Human Pose Detection & Matting System

## 项目简介 / Overview

实时人体骨骼检测和抠像系统，支持GPU加速。从网络摄像头画面中自动识别并跟踪最完整、最居中的目标人物。

Real-time human pose detection and matting system with GPU acceleration. Detects and tracks the most complete, centered person in a configurable ROI from webcam feed.

## 核心功能 / Features

- ✓ **实时摄像头捕获**: 可配置的有效区域（ROI）裁剪
- ✓ **多人骨骼检测**: 使用MediaPipe Pose检测人体关键点
- ✓ **智能人物选择**: 自动选择最完整、最高、最居中的人物
- ✓ **人体抠像**: GPU加速的实时抠像（当前版本使用简化算法）
- ✓ **实时可视化**: 四面板显示原始画面、ROI、骨骼、抠像结果

## 系统要求 / Requirements

- Windows 10/11
- Python 3.9+
- NVIDIA GPU（支持CUDA）
- CUDA Toolkit 11.8+ 和 cuDNN
- 网络摄像头

## 快速开始 / Quick Start

### 1. 安装CUDA和PyTorch

请参考 [INSTALL.md](INSTALL.md) 中的详细安装步骤。

```bash
# 安装PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 安装项目依赖

```bash
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python test_dependencies.py
```

### 4. 运行程序

**一键启动（推荐）**：
```bash
# Windows
start_rvm.bat 或 start_rvm.ps1

# 或命令行
python run_with_rvm.py
```

**原版启动**：
```bash
python src/main.py
```

💡 **提示**：RVM增强版（`run_with_rvm.py`）提供完整的调试信息和键盘控制功能，更适合开发和调试。

详细使用说明请参考：
- [RUN_RVM_README.md](RUN_RVM_README.md) - RVM增强版使用指南
- [QUICKSTART.md](QUICKSTART.md) - 快速开始

## 配置文件 / Configuration

编辑 `config.yaml` 调整设置：

### 摄像头设置
```yaml
camera:
  device_id: 0      # 摄像头ID
  resolution: [1280, 720]  # 分辨率
  fps: 30           # 帧率
```

### ROI有效区域
```yaml
roi:
  x_min: 0.4  # 左边界（归一化，0.0-1.0）
  x_max: 0.6  # 右边界
  y_min: 0.0  # 上边界
  y_max: 0.8  # 下边界
```

### 人物选择权重
```yaml
person_selection:
  completeness_weight: 0.4  # 完整性权重
  height_weight: 0.3        # 高度权重
  centeredness_weight: 0.3  # 居中度权重
```

## 使用说明 / Usage

运行后会出现4个显示面板：
1. **Original**: 原始画面 + ROI标记
2. **ROI**: 提取的有效区域
3. **Pose**: 人体骨骼检测结果
4. **Matting**: 人体抠像结果

**键盘控制：**
- `Q`: 退出程序
- `S`: 保存当前帧
- `R`: 切换ROI显示

## 性能 / Performance

- **预期帧率**: 15-20 FPS（NVIDIA GPU）
- **优化重点**: 准确性优先于速度
- **GPU加速**: PyTorch + CUDA推理

## 项目结构 / Project Structure

```
AIPoseMatch/
├── src/                         # 源代码目录
│   ├── config.py               # 配置管理
│   ├── camera_manager.py       # 摄像头和ROI
│   ├── pose_detector.py        # 骨骼检测
│   ├── person_selector.py      # 人物选择逻辑
│   ├── human_matting.py        # 人体抠像
│   ├── rvm_model.py            # RVM模型集成
│   ├── visualizer.py           # 可视化
│   └── main.py                 # 主程序（原版）
├── run_with_rvm.py             # RVM增强版启动程序 ⭐推荐
├── start_rvm.bat               # Windows一键启动
├── start_rvm.ps1               # PowerShell一键启动
├── download_rvm_model.py       # RVM模型下载工具
├── setup_rvm.sh                # RVM设置脚本
├── config.yaml                 # 配置文件
├── requirements.txt            # 依赖列表
├── test_dependencies.py        # 依赖检查脚本
├── README.md                   # 项目介绍
├── INSTALL.md                  # 安装指南
├── QUICKSTART.md               # 快速开始
├── USAGE.md                    # 使用手册
├── RUN_RVM_README.md           # RVM版本使用指南 ⭐
├── RVM_INTEGRATION.md          # RVM集成说明
└── 快速开始.txt                # 快速参考
```

## 技术栈 / Tech Stack

- **Python 3.9+**: 主要编程语言
- **MediaPipe Pose**: 人体骨骼检测
- **OpenCV**: 图像处理和摄像头
- **PyTorch**: 深度学习框架（CUDA加速）
- **NumPy**: 数值计算
- **YAML**: 配置文件

## 注意事项 / Notes

⚠️ **抠像算法**：
- 默认使用简化的HSV颜色空间抠像算法（快速可用）
- RVM框架已集成，支持加载官方RVM模型获得更好效果
- 详见 [RVM_INTEGRATION.md](RVM_INTEGRATION.md) 了解如何启用完整RVM

## 故障排除 / Troubleshooting

详见 [INSTALL.md](INSTALL.md) 和 [QUICKSTART.md](QUICKSTART.md)

常见问题：
1. CUDA不可用 → 检查PyTorch CUDA版本匹配
2. 摄像头无法打开 → 修改device_id或检查占用
3. 检测不到人体 → 调整光照或降低检测阈值

## License

MIT License

