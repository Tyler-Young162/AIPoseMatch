# 🎯 Unity通信正确启动步骤

## ❌ 错误理解
不要运行 `run_dual_camera.py` - 它没有Unity通信功能

## ✅ 正确步骤

### 步骤1：启动Python后台服务

**方式1：带调试窗口（推荐用于测试）**
```bash
.venv\Scripts\python.exe run_backend_service.py --show-window
```

**方式2：不带窗口（纯后台）**
```bash
.venv\Scripts\python.exe run_backend_service.py
```

### 步骤2：启动Unity

1. 在Unity Editor中打开项目：`unity/AIPoseUnity`
2. 点击Play按钮运行场景
3. 查看Console，应该看到：
   - "已连接到Python服务器 127.0.0.1:8888"

### 步骤3：验证数据流

- ✅ Unity Console显示"已连接到Python服务器"
- ✅ Python控制台显示"Unity客户端已连接到视频流端口"
- ✅ Unity场景中显示摄像头画面
- ✅ 评分数字实时更新

## 📊 程序对比

| 程序 | 功能 | Unity通信 | 使用场景 |
|------|------|-----------|---------|
| `run_dual_camera.py` | 双摄像头本地显示 | ❌ 无 | 本地测试、调试 |
| `run_backend_service.py` | 双摄像头+Unity推流 | ✅ 有 | Unity集成、最终使用 |
| `run_with_rvm.py` | 单摄像头本地显示 | ❌ 无 | 单摄像头测试 |
| `run_with_tray.py` | 系统托盘+可选Unity | ✅ 可选 | 后台运行 |

## 🎯 关键点

**只有 `run_backend_service.py` 包含Unity通信代码**

其他程序都不包含Unity通信模块，如果运行它们，Unity无法接收数据。

## ✅ 验证清单

运行 `run_backend_service.py` 后，检查Python控制台：
- ✅ "初始化Unity通信..."
- ✅ "视频流服务器已启动，等待Unity连接... 127.0.0.1:8888"
- ✅ "控制服务器监听在 127.0.0.1:8889"
- ✅ "后台服务开始运行..."

然后启动Unity，应该看到：
- ✅ "Unity客户端已连接到视频流端口"

## 🎉 成功标志

当看到以下情况时，说明一切正常：
1. ✅ Unity Console："已连接到Python服务器"
2. ✅ Python Console："Unity客户端已连接到视频流端口"
3. ✅ Unity场景显示实时画面
4. ✅ 评分实时更新
5. ✅ 姿态切换按钮工作正常

