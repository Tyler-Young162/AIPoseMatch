# 测试Unity连接

## ✅ 修复完成

已修复端口冲突问题：
- ✅ Unity不再启动控制服务器
- ✅ Python监听8888和8889端口
- ✅ Unity连接到Python发送命令

## 📝 测试步骤

### 1. 启动Python后端

```bash
.venv\Scripts\python.exe run_backend_service.py --show-window
```

**预期输出：**
```
[Unity通信] 视频流服务器已启动，等待Unity连接... 127.0.0.1:8888
[Unity通信] 控制服务器监听在 127.0.0.1:8889
初始化完成！
后台服务开始运行...
```

### 2. 启动Unity

在Unity Editor中点击Play按钮

**Unity Console预期输出：**
```
[PoseMatchReceiver] 开始接收数据...
视频端口: 8888, 控制端口: 8889
[PoseMatchReceiver] 视频流：连接到Python 127.0.0.1:8888
[PoseMatchReceiver] 控制命令：将连接到Python 127.0.0.1:8889
[PoseMatchReceiver] 已连接到Python服务器 127.0.0.1:8888
```

**不应该再出现端口冲突错误！**

### 3. 验证连接

Python Console应该显示：
```
[Unity通信] Unity客户端已连接到视频流端口: ('127.0.0.1', xxxxx)
```

### 4. 测试控制命令

在Unity中点击"下一个姿态"或"上一个姿态"按钮

Python Console应该显示：
```
[Unity通信] Unity客户端已连接: ('127.0.0.1', xxxxx)
[Unity通信] 收到pose切换命令: ...
```

## ✅ 成功标志

- ✅ 无端口冲突错误
- ✅ Unity连接成功
- ✅ Python收到连接
- ✅ 控制命令正常传递

