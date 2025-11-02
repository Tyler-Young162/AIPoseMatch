# Unity通信测试计划

## 目标
测试Python后台服务与Unity之间的实时视频流通信

## 测试步骤

### 第一步：检查Python后端
1. ✅ Python后端启动成功
2. ✅ 摄像头初始化成功  
3. ✅ Unity通信模块加载成功

### 第二步：运行后台服务（不带窗口）
```bash
.venv\Scripts\python.exe run_backend_service.py
```

预期结果：
- Unity连接失败（因为Unity未启动）
- 显示 "[WARN] 无法连接到Unity，将在无Unity连接模式下运行"
- 后台继续运行

### 第三步：运行后台服务（带调试窗口）
```bash
.venv\Scripts\python.exe run_backend_service.py --show-window
```

预期结果：
- 显示两个调试窗口（摄像头0和摄像头1）
- 可以查看实时画面
- 验证姿态比对功能

### 第四步：Unity端测试
1. 检查Unity项目配置
2. 确认PoseMatchReceiver脚本已添加
3. 启动Unity场景
4. 查看连接状态

### 第五步：完整集成测试
1. 启动Python后端
2. 启动Unity
3. 验证视频流接收
4. 测试姿态切换按钮

## 文件清单

### Python端
- `run_backend_service.py` - 后台服务主程序
- `src/unity_communication.py` - Unity通信模块
- `src/pose_matcher.py` - 姿态比对模块
- `src/human_matting.py` - 抠像模块

### Unity端
- `unity/AIPoseUnity/Assets/Scripts/PoseMatchReceiver.cs` - Unity接收脚本

## 通信协议

### Python -> Unity（端口8888）
```
[4字节] "FRAM" 帧头
[4字节] 元数据长度 (uint32, 大端)
[4字节] 图像长度 (uint32, 大端)
[变长] JSON元数据
[变长] JPEG图像数据
```

### Unity -> Python（端口8889）
```
[4字节] JSON长度 (uint32, 大端)
[变长] JSON命令
```

## 当前状态
准备开始第一步测试

