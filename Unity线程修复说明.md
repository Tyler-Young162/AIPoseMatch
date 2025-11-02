# Unity主线程纹理更新修复

## 🐛 问题描述

Unity报错：
```
SupportsTextureFormatNative can only be called from the main thread
```

## 🔍 根本原因

在后台线程（`ReceiveVideoThread`）中直接调用了：
1. `new Texture2D(2, 2)` - 创建纹理对象
2. `texture.LoadImage(imageBytes)` - 加载图像数据

这些Unity API只能在**主线程**中调用！

## ✅ 解决方案

使用**队列机制**，将纹理创建工作从后台线程转移到主线程：

### 修改内容

1. **添加队列**（第51行）
   ```csharp
   private Queue<byte[]> textureUpdateQueue = new Queue<byte[]>();
   ```

2. **后台线程：只接收数据**（第154-158行）
   ```csharp
   // 将图像数据放入队列，等待主线程处理
   lock (lockObject)
   {
       textureUpdateQueue.Enqueue(imageBytes);
   }
   ```

3. **主线程：处理纹理**（第326-336行）
   ```csharp
   void Update()
   {
       // 处理纹理更新队列（主线程执行）
       lock (lockObject)
       {
           if (textureUpdateQueue.Count > 0)
           {
               byte[] imageBytes = textureUpdateQueue.Dequeue();
               UpdateTexture(imageBytes);  // 在主线程创建和加载
           }
       }
       // ... UI更新
   }
   ```

### 执行流程

```
后台线程（ReceiveVideoThread）
  ↓ 接收网络数据
  ↓ 解析帧头和元数据
  ↓ 读取图像字节流
  ↓ 放入队列（Enqueue）
  
主线程（Update每帧）
  ↓ 检查队列
  ↓ 取出数据（Dequeue）
  ↓ 创建Texture2D ← 主线程！
  ↓ 调用LoadImage() ← 主线程！
  ↓ 更新材质显示
```

## 🧹 清理

同时移除了无用的代码：
- `ControlServerThread()` 函数（已删除）
- `controlServer` 变量
- `controlThread` 变量

## ✅ 测试

现在可以正常测试：
1. 运行 `run_backend_service.py --show-window`
2. Unity中点击Play
3. 应该看到画面，无报错

