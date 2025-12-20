"""
GPU 模块：GPU 初始化、配置和工具函数
延迟加载设计，避免子进程中重复初始化
"""

import os

# 全局变量（函数内初始化）
USE_GPU = False
GPU_NAME = None
GPU_MEMORY = 0
DEVICE = None
torch = None
_GPU_INITIALIZED = False

def initialize_gpu():
    """初始化GPU配置（只在主进程中执行一次）"""
    global USE_GPU, GPU_NAME, GPU_MEMORY, DEVICE, torch, _GPU_INITIALIZED
    
    if _GPU_INITIALIZED:
        return
    
    _GPU_INITIALIZED = True
    
    print(f"✅ GPU检测...")
    
    try:
        import torch as torch_module
        torch = torch_module
        
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        if torch.cuda.is_available():
            USE_GPU = True
            DEVICE = torch.device('cuda:0')
            GPU_NAME = torch.cuda.get_device_name(0)
            GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   🚀 检测到GPU: {GPU_NAME} ({GPU_MEMORY:.1f}GB)")
            print(f"   💡 将使用GPU 0进行计算加速")
            print(f"   🚀 GPU优化策略:")
            print(f"      🚀 表达式树编译为PyTorch计算图（2.8-4.6倍加速）")
            print(f"      1. 数据预加载到GPU内存（一次性转换）")
            print(f"      2. 表达式树编译为lambda函数（零Python递归开销）")
            print(f"      3. PyTorch自动算子融合（减少显存访问）")
            print(f"      4. ATR/MA计算向量化（避免Python循环）")
        else:
            USE_GPU = False
            DEVICE = torch.device('cpu')
            print(f"   ⚠️  未检测到GPU，将使用CPU模式")
    except ImportError:
        USE_GPU = False
        DEVICE = None
        torch = None
        print("   ⚠️  PyTorch未安装，使用CPU模式")
        print("   💡 建议安装PyTorch加速：pip install torch")


def to_tensor(arr):
    """数组转张量"""
    global torch, DEVICE
    if torch is None:
        try:
            import torch as torch_module
            torch = torch_module
        except ImportError:
            pass
    
    if USE_GPU and torch is not None and DEVICE is not None:
        return torch.tensor(arr, dtype=torch.float32, device=DEVICE)
    return arr


def to_numpy(tensor):
    """张量转数组"""
    global torch
    if torch is None:
        try:
            import torch as torch_module
            torch = torch_module
        except ImportError:
            pass
    
    if USE_GPU and torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    return tensor
