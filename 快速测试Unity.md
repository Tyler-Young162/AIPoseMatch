# 🚀 快速测试Unity通信

## ✅ 正确步骤

### 第1步：启动Python后端

**使用虚拟环境运行：**
```bash
.venv\Scripts\python.exe run_backend_service.py --show-window
```

或者使用启动脚本：
```bash
# PowerShell
.\start_tray.ps1
```

**预期输出：**
```
[Unity通信] 视频流服务器已启动，等待Unity连接... 127.0.0.1:8888
[Unity通信] 控制服务器监听在 127.0.0.1:8889
初始化完成！
后台服务开始运行...
```

### 第2步：启动Unity

1. 打开Unity Editor
2. 打开项目：`unity/AIPoseUnity`
3. 点击Play按钮

**Unity Console应该显示：**
```
[PoseMatchReceiver] 开始接收数据...
视频端口: 8888, 控制端口: 8889
[PoseMatchReceiver] 控制服务器已启动,监听 127.0.0.1:8889
[PoseMatchReceiver] 已连接到Python服务器 127.0.0.1:8888
```

### 第3步：验证数据流

如果连接成功：
- ✅ Unity Console：显示"已连接到Python服务器"
- ✅ Python Console：显示"Unity客户端已连接到视频流端口"
- ✅ Unity场景：显示摄像头画面（如果有Material设置）
- ✅ 评分：实时更新数字

### 第4步：测试控制命令

在Unity中：
1. 点击"下一个姿态"按钮
2. 查看Python Console是否收到命令
3. 查看Unity Console是否显示发送日志

## ⚠️ 常见问题

### Python卡住不启动
可能原因：摄像头初始化问题
解决：确保有2个摄像头，或检查摄像头是否被占用

### Unity连接被拒绝
可能原因：Python还没启动
解决：先启动Python，再启动Unity

### 连接成功但看不到画面
可能原因：Unity Material未设置或纹理更新逻辑问题
解决：检查Unity Scene中的Material和Script设置

## 🎯 关键文件

- **Python端**：`run_backend_service.py` - 包含Unity通信
- **Unity端**：`unity/AIPoseUnity/Assets/Scripts/PoseMatchReceiver.cs`
- **通信模块**：`src/unity_communication.py`

## 📝 重要提示

**只能使用 `run_backend_service.py` 进行Unity通信测试**

`run_dual_camera.py` 和其他程序都不包含Unity通信功能！

