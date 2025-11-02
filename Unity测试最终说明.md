# Unity通信测试 - 最终说明

## ✅ 已修复的问题

1. ✅ **托盘环境问题** - 使用虚拟环境启动
2. ✅ **Unity连接方向** - Python作为服务端
3. ✅ **Unity端口冲突** - 删除Unity的控制服务器
4. ✅ **摄像头未初始化** - 添加摄像头初始化代码

## 🚀 测试步骤

### 完整启动流程

**步骤1：启动Python后端**
```bash
.venv\Scripts\python.exe run_backend_service.py --show-window
```

**预期输出：**
- 初始化摄像头0和1
- 初始化抠像模块
- 启动Unity服务器
- 显示调试窗口

**步骤2：启动Unity**
- 在Unity Editor中点击Play
- 查看Console日志

**步骤3：验证功能**
- 查看调试窗口是否显示画面
- 验证Unity是否收到数据
- 测试控制按钮

## 📊 应该看到什么

### Python Console
```
初始化摄像头0...
  正在打开摄像头 0...
Camera initialized: 1280x720 @ 30.0 FPS
[OK] 摄像头0初始化成功

初始化摄像头1...
  正在打开摄像头 1...
Camera initialized: 1280x720 @ 30.0 FPS
[OK] 摄像头1初始化成功

[OK] 摄像头0抠像模块初始化成功
[OK] 摄像头1抠像模块初始化成功

[Unity通信] 视频流服务器已启动，等待Unity连接...
[Unity通信] 控制服务器监听在 127.0.0.1:8889

[Unity通信] Unity客户端已连接到视频流端口: ('127.0.0.1', xxxxx)

初始化完成！
后台服务开始运行...
```

### Unity Console
```
[PoseMatchReceiver] 开始接收数据...
视频端口: 8888, 控制端口: 8889
[PoseMatchReceiver] 视频流：连接到Python 127.0.0.1:8888
[PoseMatchReceiver] 已连接到Python服务器 127.0.0.1:8888
```

### 画面显示
- **Python调试窗口**：两个OpenCV窗口显示摄像头画面
- **Unity场景**：如果配置了Material，应该显示画面

## 🐛 故障排除

### 摄像头初始化失败
- 检查是否有2个摄像头
- 检查摄像头是否被占用
- 尝试重启程序

### 连接成功但无画面
- 检查调试窗口是否显示
- 检查Unity Material设置
- 查看是否有解码错误

### 评分不更新
- 检查UI绑定
- 查看是否有数据发送错误

## ✅ 成功标志

所有以下条件满足时，测试成功：
1. ✅ Python显示摄像头初始化成功
2. ✅ Unity连接成功
3. ✅ 调试窗口显示画面
4. ✅ Unity场景显示画面
5. ✅ 数据正常推送

## 📝 重要提示

**只能使用 `run_backend_service.py` 测试Unity通信**

其他程序（如`run_dual_camera.py`）不包含Unity通信功能。

