# RVM集成更新说明

## 更新内容

已成功将RVM（Robust Video Matting）框架集成到AI Pose Match项目中。

## 新增文件

### 1. 核心模块
- **src/rvm_model.py** - RVM模型定义和工具函数
  - `MattingNetwork` 类：RVM模型结构
  - `load_rvm_model()` 函数：加载模型
  - `convert_to_rgb_tensor()` 函数：图像转张量
  - `convert_from_tensor()` 函数：张量转图像

### 2. 工具脚本
- **download_rvm_model.py** - 自动下载RVM模型文件
  - 从GitHub releases下载预训练权重
  - 支持MobileNetV3和ResNet50两种变体

- **setup_rvm.sh** - Linux/Mac设置脚本
- **setup_rvm.ps1** - Windows PowerShell设置脚本
  - 克隆官方RVM仓库
  - 检查模型文件
  - 提供下载指引

### 3. 文档
- **RVM_INTEGRATION.md** - RVM集成指南
  - 完整集成步骤
  - 配置选项说明
  - 故障排除指南

## 修改文件

### src/human_matting.py
主要更新：
1. 添加RVM模块导入
2. 更新 `initialize()` 方法支持RVM模型加载
3. 添加 `_extract_rvm_matte()` 方法实现RVM推理
4. 自动回退到简化版本
5. 支持多路径模型查找

关键特性：
- 自动检测RVM模型文件
- 支持GPU加速
- 支持降采样提高性能
- 完善的错误处理

### README.md
- 更新抠像算法说明
- 添加RVM集成文档链接

## 使用方式

### 方式1：使用简化版本（默认）

无需任何操作，程序会自动使用HSV颜色空间抠像：

```bash
python src/main.py
```

### 方式2：使用完整RVM

**步骤1**：下载模型文件

```bash
python download_rvm_model.py
```

或手动下载：
```bash
mkdir models
cd models
# 下载到models目录
```

**步骤2**：运行程序

模型文件存在时会自动使用RVM：

```bash
python src/main.py
```

### 配置选项

在 `config.yaml` 中调整：

```yaml
matting:
  model: "rvm"  # 模型类型
  downsample_ratio: 0.25  # 降采样（0.25-1.0）
  filter_incomplete: true  # 过滤不完全人物
  min_person_height_ratio: 0.3  # 最小人物高度
```

## 技术细节

### 模型架构

简化版RVM架构：
- **骨干网络**：MobileNetV3或ResNet50
- **特征提取**：多尺度特征
- **解码器**：卷积层生成alpha和前景
- **输出**：alpha遮罩和前景图像

### 推理流程

1. BGR图像转RGB张量
2. 可选降采样（性能优化）
3. 模型前向传播
4. 上采样回原尺寸
5. 转换为alpha遮罩

### 性能优化

- **降采样**：可设置downsample_ratio提高速度
- **GPU加速**：自动使用CUDA
- **批处理**：支持批量推理（未来）
- **量化**：模型量化（未来）

## 当前限制

1. **简化实现**：当前提供的是简化版RVM
   - 可能与官方模型权重不完全兼容
   - 建议集成完整RVM获得最佳效果

2. **时序一致性**：未实现视频稳像
   - 当前按帧独立处理
   - 可能在某些场景下有抖动

3. **多尺度**：特征融合较简单
   - 完整RVM有更复杂的多尺度架构

## 改进建议

### 短期改进
1. 添加时序缓存改善稳定性
2. 优化解码器结构
3. 添加更多后处理选项

### 长期改进
1. 集成官方完整RVM实现
2. 支持模型训练
3. 添加背景替换功能
4. 支持更多模型变体

## 测试建议

### 功能测试
1. ✅ 简化版本工作正常
2. ⚠️ RVM模型加载（需要下载文件）
3. ⚠️ GPU加速功能
4. ⚠️ 降采样性能优化

### 性能测试
建议测试项：
- 不同downsample_ratio下的帧率
- GPU vs CPU性能对比
- 不同分辨率下的表现
- 内存使用情况

### 质量测试
建议测试项：
- 边缘精确度
- 复杂背景处理
- 光照变化适应性
- 快速运动处理

## 参考资源

- RVM官方仓库：https://github.com/PeterL1n/RobustVideoMatting
- 论文：Robust High-Resolution Video Matting with Temporal Guidance
- 模型下载：https://github.com/PeterL1n/RobustVideoMatting/releases

## 更新日志

### v1.1 (当前)
- ✅ 添加RVM框架集成
- ✅ 实现简化版模型
- ✅ 添加工具脚本
- ✅ 完善文档

### v1.0 (之前)
- ✅ 基础人体检测和抠像
- ✅ HSV颜色空间抠像
- ✅ 完整项目结构

---

**注意**：RVM是一个研究级别的项目，使用时请遵循其许可证要求。

