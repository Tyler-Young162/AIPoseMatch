"""
独立脚本：分析RVM权重文件结构，诊断为什么模型无法正常工作
"""
import torch
from pathlib import Path

def analyze_checkpoint(checkpoint_path: str):
    """分析RVM checkpoint文件的结构"""
    print("=" * 70)
    print("RVM权重文件分析工具")
    print("=" * 70)
    
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"错误: 文件不存在: {checkpoint_path}")
        return
    
    print(f"\n文件路径: {checkpoint_path}")
    file_size = path.stat().st_size / (1024 * 1024)  # MB
    print(f"文件大小: {file_size:.1f} MB")
    
    try:
        print("\n正在加载checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 检查checkpoint结构
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("✓ Checkpoint包含 'state_dict'")
        else:
            state_dict = checkpoint
            print("✓ Checkpoint直接是state_dict")
        
        total_keys = len(state_dict.keys())
        print(f"\n总键数: {total_keys}")
        
        # 分析键名模式
        print("\n" + "=" * 70)
        print("键名模式分析")
        print("=" * 70)
        
        key_groups = {}
        for key in state_dict.keys():
            # 按第一个点分割，获取主要组件名
            parts = key.split('.')
            if len(parts) > 0:
                component = parts[0]
                if component not in key_groups:
                    key_groups[component] = []
                key_groups[component].append(key)
        
        # 按组件分组显示
        for component in sorted(key_groups.keys()):
            keys = key_groups[component]
            print(f"\n[{component}] - {len(keys)} 个键")
            # 显示前5个示例
            for i, key in enumerate(keys[:5]):
                shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
                print(f"  {i+1}. {key} -> shape: {shape}")
            if len(keys) > 5:
                print(f"  ... 还有 {len(keys) - 5} 个键")
        
        # 详细列出所有键（前50个）
        print("\n" + "=" * 70)
        print("所有键列表（前50个）")
        print("=" * 70)
        for i, key in enumerate(list(state_dict.keys())[:50]):
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            dtype = state_dict[key].dtype if hasattr(state_dict[key], 'dtype') else 'N/A'
            print(f"{i+1:3d}. {key:<60} {str(shape):<20} {str(dtype)}")
        
        if total_keys > 50:
            print(f"\n... 还有 {total_keys - 50} 个键未显示")
        
        # 检查关键组件
        print("\n" + "=" * 70)
        print("关键组件检查")
        print("=" * 70)
        
        expected_components = ['backbone', 'decoder', 'refiner', 'downsample_conv', 'downsample_bn']
        for comp in expected_components:
            matching_keys = [k for k in state_dict.keys() if k.startswith(comp)]
            if matching_keys:
                print(f"✓ 找到 '{comp}' 组件: {len(matching_keys)} 个键")
                print(f"  示例: {matching_keys[0]}")
            else:
                print(f"✗ 未找到 '{comp}' 组件")
        
        # 检查是否有ConvGRU相关键
        gru_keys = [k for k in state_dict.keys() if 'gru' in k.lower() or 'rnn' in k.lower()]
        if gru_keys:
            print(f"\n✓ 找到时序相关组件: {len(gru_keys)} 个键")
            print(f"  示例: {gru_keys[0]}")
        else:
            print(f"\n✗ 未找到时序相关组件（ConvGRU/RNN）")
            print("  注意: 官方RVM使用ConvGRU，我们的简化实现没有")
        
        print("\n" + "=" * 70)
        print("诊断结论")
        print("=" * 70)
        
        # 检查我们的模型是否有这些键
        print("\n我们简化实现中的键名:")
        simplified_keys = [
            'backbone.features',  # MobileNetV3
            'decoder.0',  # Conv2d
            'decoder.1',  # BatchNorm
            'decoder.2',  # ReLU
            # ... 等等
        ]
        
        # 检查匹配度
        backbone_keys = [k for k in state_dict.keys() if 'backbone' in k]
        decoder_keys = [k for k in state_dict.keys() if 'decoder' in k]
        
        print(f"\nCheckpoint中的backbone键: {len(backbone_keys)} 个")
        if backbone_keys:
            print(f"  示例: {backbone_keys[0]}")
        
        print(f"\nCheckpoint中的decoder键: {len(decoder_keys)} 个")
        if decoder_keys:
            print(f"  前3个示例:")
            for key in decoder_keys[:3]:
                print(f"    - {key}")
        
        # 检查是否有refiner
        refiner_keys = [k for k in state_dict.keys() if 'refiner' in k.lower()]
        if refiner_keys:
            print(f"\n✓ Checkpoint包含refiner组件: {len(refiner_keys)} 个键")
            print("  注意: 我们的简化实现没有refiner，这会导致权重不匹配")
        
        print("\n" + "=" * 70)
        print("建议")
        print("=" * 70)
        print("\n根据分析结果，建议:")
        print("1. 如果权重匹配率 < 50%，说明架构不兼容")
        print("2. 需要查看官方RVM实现来了解正确的架构")
        print("3. 或者直接使用官方RVM代码而不是简化实现")
        print("=" * 70)
        
    except Exception as e:
        import traceback
        print(f"\n错误: {e}")
        print("\n详细错误:")
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    # 默认检查models目录
    model_paths = [
        "models/rvm_mobilenetv3.pth",
        "models/rvm_resnet50.pth",
    ]
    
    if len(sys.argv) > 1:
        model_paths = [sys.argv[1]]
    
    found = False
    for model_path in model_paths:
        if Path(model_path).exists():
            analyze_checkpoint(model_path)
            found = True
            break
    
    if not found:
        print("未找到RVM模型文件。请先下载:")
        print("  python download_rvm_model.py")
        print("\n或手动指定路径:")
        print("  python analyze_rvm_weights.py <模型文件路径>")

