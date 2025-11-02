# Unity完整测试指南

## ✅ 已修复的问题

1. ✅ Python连接Unity失败 → 改为Python作为服务端
2. ✅ Unity端口冲突 → 删除Unity的控制服务器

## 🚀 完整测试步骤

### 步骤1：启动Python后端

```bash
.venv\Scripts\python.exe run_backend_service.py --show-window
```

**预期输出：**
```
AI Pose Match - 后台服务模式
初始化Unity通信...
[Unity通信] 视频流服务器已启动，等待Unity连接... 127.0.0.1:8888
[Unity通信] 控制服务器监听在 127.0.0.1:8889
初始化姿态比对模块...
[OK] 已加载 6 个目标姿态
初始化摄像头0...
初始化摄像头1...
初始化完成！
后台服务开始运行...
```

### 步骤2：启动Unity

1. 在Unity Editor中打开项目
2. 点击Play按钮

**Unity Console预期输出：**
```
[PoseMatchReceiver] 开始接收数据...
视频端口: 8888, 控制端口: 8889
[PoseMatchReceiver] 视频流：连接到Python 127.0.0.1:8888
[PoseMatchReceiver] 控制命令：将连接到Python 127.0.0.1:8889
[PoseMatchReceiver] 已连接到Python服务器 127.0.0.1:8888
```

**Python Console应该显示：**
```
[Unity通信] Unity客户端已连接到视频流端口: ('127.0.0.1', xxxxx)
```

### 步骤3：验证功能

#### 3.1 查看调试窗口
- 应该看到两个OpenCV窗口
- 显示实时的摄像头画面和骨骼

#### 3.2 验证Unity画面
- 如果Unity场景中设置了Material
- 应该看到摄像头画面

#### 3.3 测试评分
- 查看Unity UI中的评分Text
- 应该实时更新分数

#### 3.4 测试姿态切换
1. 在Unity中点击"下一个姿态"按钮
2. Python Console应该显示：
   ```
   [Unity通信] Unity客户端已连接: ('127.0.0.1', xxxxx)
   [Unity通信] 收到pose切换命令: ...
   [Unity] 切换到下一个姿态: poseB
   ```
3. Unity画面应该切换到新的姿态

## 🐛 故障排除

### Unity连接失败
- 确保Python先启动
- 检查端口是否被占用

### 看不到画面
- 检查Unity Material设置
- 检查纹理更新逻辑

### 控制按钮无响应
- 检查按钮回调设置
- 查看Python Console是否有命令接收日志

### 评分不更新
- 检查UI Text组件绑定
- 查看Python数据是否正常发送

## ✅ 成功标准

当满足以下所有条件时，表示测试成功：
1. ✅ Unity无错误连接
2. ✅ Python显示连接成功
3. ✅ 调试窗口显示画面
4. ✅ Unity场景显示画面（如果配置）
5. ✅ 评分实时更新
6. ✅ 姿态切换按钮工作

## 📝 测试命令

完整的一条龙测试：

```bash
# 终端1：启动Python
.venv\Scripts\python.exe run_backend_service.py --show-window

# 终端2：查看日志
# 查看上面的Python输出

# Unity Editor：启动场景
# 点击Play按钮，查看Console

# 测试交互
# 在Unity中点击"下一个姿态"按钮
# 查看两个Console的日志
```

