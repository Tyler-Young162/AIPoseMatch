# RVM (Robust Video Matting) 集成指南

## 概述

当前项目已经为RVM集成做好了准备。目前实现的RVM模型是一个简化版本，提供了基本框架。为了获得最佳效果，您可以选择：

1. **使用简化版本**（当前默认）- 基于HSV颜色空间，快速但效果一般
2. **集成完整RVM** - 下载官方模型并使用完整实现

## 选项1：使用简化版本（默认）

当前实现会自动回退到HSV颜色空间抠像算法。这是最快的方案，无需额外下载。

### 优点
- 无需下载大型模型文件
- 运行速度快
- 对于均匀背景效果可接受

### 缺点
- 抠像质量有限
- 边缘可能不够精确
- 需要良好的光照条件

## 选项2：集成完整RVM

### 步骤1：下载模型文件

创建 `models` 目录并下载模型文件：

```bash
mkdir models
cd models

# 下载轻量级模型（推荐）
wget https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0/rvm_mobilenetv3.pth

# 或下载高质量模型（更大，更慢）
wget https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0/rvm_resnet50.pth
```

或者使用提供的下载脚本：

```bash
python download_rvm_model.py
```

### 步骤2：克隆官方RVM仓库

为了使用完整的RVM实现，需要克隆官方仓库：

```bash
git clone https://github.com/PeterL1n/RobustVideoMatting.git
```

### 步骤3：修改模型加载代码

由于官方RVM的实现较复杂，当前提供的简化版本可能无法直接加载官方权重。

**推荐的完整集成方案**：

1. 在 `src/human_matting.py` 中添加对官方RVM的导入：

```python
try:
    import sys
    import os
    rvm_path = os.path.join(os.path.dirname(__file__), '..', 'RobustVideoMatting')
    if os.path.exists(rvm_path):
        sys.path.insert(0, rvm_path)
        from inference import convert_video, convert_image
        RVM_OFFICIAL_AVAILABLE = True
    else:
        RVM_OFFICIAL_AVAILABLE = False
except:
    RVM_OFFICIAL_AVAILABLE = False
```

2. 使用官方RVM模型进行推理。

## 当前实现状态

### 已实现功能
- ✅ RVM模型加载框架
- ✅ 张量转换工具
- ✅ 自动降采样支持
- ✅ 简化版本回退
- ✅ GPU加速支持

### 待完善功能
- ⚠️ 完整的RVM模型结构（当前为简化版）
- ⚠️ 官方模型权重的正确加载
- ⚠️ 时序一致性（视频稳像）
- ⚠️ 多尺度特征提取

## 建议方案

对于您的使用场景，我建议：

### 短期方案（立即可用）
使用当前实现的简化版本，它已经可以工作：

1. HSV颜色空间抠像
2. 基本的形态学操作
3. 不完全人物过滤

### 长期方案（获得最佳效果）
集成完整的官方RVM：

1. 克隆官方仓库
2. 下载预训练模型
3. 使用官方推理接口
4. 根据需要修改集成代码

## 性能对比

| 方案 | 速度 | 质量 | 资源占用 |
|------|------|------|----------|
| 简化版本 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| RVM MobileNetV3 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| RVM ResNet50 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## 代码使用

无论使用哪种方案，代码接口都是一致的：

```python
from src.human_matting import HumanMatting
from config import Config

config = Config.load_from_yaml("config.yaml")
matting = HumanMatting(config)
matting.initialize()

# 处理图像
alpha, result = matting.process(frame)
```

如果RVM模型可用，会自动使用；否则回退到简化版本。

## 配置选项

在 `config.yaml` 中配置：

```yaml
matting:
  model: "rvm"  # 使用RVM或"simple"
  downsample_ratio: 0.25  # 降采样比例（速度vs质量）
  filter_incomplete: true  # 过滤不完全人物
  min_person_height_ratio: 0.3  # 最小人物高度
```

## 故障排除

### 模型文件未找到

```
Warning: RVM model file not found. Please download it using:
  python download_rvm_model.py
```

**解决**：下载模型文件到 `models/` 目录

### CUDA不可用

```
Human matting device: cpu
```

**解决**：确保安装GPU版本的PyTorch

### 加载失败自动回退

程序会自动回退到简化版本，无需操作。查看控制台输出了解详细信息。

## 更多信息

- RVM官方仓库：https://github.com/PeterL1n/RobustVideoMatting
- 模型下载：https://github.com/PeterL1n/RobustVideoMatting/releases
- 论文：Robust High-Resolution Video Matting with Temporal Guidance

## 总结

当前项目已经提供了RVM集成的完整框架。默认使用简化版本可立即使用，如需更好效果可按本文档集成完整RVM。

