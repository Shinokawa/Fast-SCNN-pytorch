"""
测试优化后的训练配置
"""
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time

from utils.loss import DiceLoss, MixDiceLoss, FocalDiceLoss
from models.fast_scnn import get_fast_scnn

def test_loss_functions():
    """测试新的损失函数"""
    print("Testing Loss Functions...")
    
    # 创建模拟数据
    batch_size, height, width = 4, 480, 480
    num_classes = 2
    
    # 模拟预测和目标
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred = torch.randn(batch_size, num_classes, height, width).to(device)
    target = torch.randint(0, num_classes, (batch_size, height, width)).to(device)
    
    # 测试 Dice Loss
    print("\n1. Testing Dice Loss...")
    dice_loss = DiceLoss().to(device)
    start_time = time.time()
    for _ in range(10):
        loss = dice_loss(pred, target)
    dice_time = time.time() - start_time
    print(f"   Dice Loss: {loss.item():.4f}, Time: {dice_time:.4f}s")
    
    # 测试 Mixed Dice Loss
    print("\n2. Testing Mixed Dice Loss...")
    mix_dice_loss = MixDiceLoss(aux=True).to(device)
    aux_pred = torch.randn(batch_size, num_classes, height, width).to(device)
    start_time = time.time()
    for _ in range(10):
        loss = mix_dice_loss((pred, aux_pred), target)
    mix_dice_time = time.time() - start_time
    print(f"   Mixed Dice Loss: {loss.item():.4f}, Time: {mix_dice_time:.4f}s")
    
    # 测试 Focal Dice Loss
    print("\n3. Testing Focal Dice Loss...")
    focal_dice_loss = FocalDiceLoss().to(device)
    start_time = time.time()
    for _ in range(10):
        loss = focal_dice_loss(pred, target)
    focal_dice_time = time.time() - start_time
    print(f"   Focal Dice Loss: {loss.item():.4f}, Time: {focal_dice_time:.4f}s")
    
    print(f"\nPerformance Comparison (10 iterations):")
    print(f"   Dice Loss:       {dice_time:.4f}s")
    print(f"   Mixed Dice Loss: {mix_dice_time:.4f}s")
    print(f"   Focal Dice Loss: {focal_dice_time:.4f}s")

def test_fp16_training():
    """测试FP16混合精度训练"""
    print("\n" + "="*50)
    print("Testing FP16 Mixed Precision Training...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping FP16 test")
        return
    
    device = torch.device('cuda')
    
    # 创建模型
    model = get_fast_scnn('tusimple', aux=True).to(device)
    model.train()
    
    # 创建优化器和scaler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = GradScaler()
    
    # 创建损失函数
    criterion = MixDiceLoss(aux=True).to(device)
    
    # 模拟数据
    batch_size = 16
    images = torch.randn(batch_size, 3, 480, 480).to(device)
    targets = torch.randint(0, 2, (batch_size, 480, 480)).to(device)
    
    print(f"\nTesting with batch size: {batch_size}")
    
    # 测试FP32训练
    print("\n1. FP32 Training...")
    model.train()
    start_time = time.time()
    for i in range(5):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if i == 0:
            print(f"   First iteration loss: {loss.item():.4f}")
    fp32_time = time.time() - start_time
    print(f"   5 iterations time: {fp32_time:.4f}s")
    
    # 测试FP16训练
    print("\n2. FP16 Training...")
    model.train()
    start_time = time.time()
    for i in range(5):
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if i == 0:
            print(f"   First iteration loss: {loss.item():.4f}")
    fp16_time = time.time() - start_time
    print(f"   5 iterations time: {fp16_time:.4f}s")
    
    speedup = fp32_time / fp16_time
    print(f"\nSpeedup with FP16: {speedup:.2f}x")
    
    # 内存使用检查
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
    print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")

def test_batch_size_scaling():
    """测试不同batch size的性能"""
    print("\n" + "="*50)
    print("Testing Batch Size Scaling...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping batch size test")
        return
    
    device = torch.device('cuda')
    model = get_fast_scnn('tusimple', aux=True).to(device)
    criterion = MixDiceLoss(aux=True).to(device)
    
    batch_sizes = [4, 8, 16, 32] if torch.cuda.get_device_properties(0).total_memory > 8e9 else [4, 8, 16]
    
    for batch_size in batch_sizes:
        try:
            print(f"\nTesting batch size: {batch_size}")
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            
            images = torch.randn(batch_size, 3, 480, 480).to(device)
            targets = torch.randint(0, 2, (batch_size, 480, 480)).to(device)
            
            model.train()
            start_time = time.time()
            
            # 使用FP16
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            forward_time = time.time() - start_time
            
            memory_used = torch.cuda.memory_allocated(device) / 1024**3
            print(f"   Forward time: {forward_time:.4f}s")
            print(f"   Memory used: {memory_used:.2f}GB")
            print(f"   Loss: {loss.item():.4f}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   OOM at batch size {batch_size}")
                break
            else:
                raise e

if __name__ == '__main__':
    print("="*60)
    print("Performance Optimization Test")
    print("="*60)
    
    # 测试损失函数
    test_loss_functions()
    
    # 测试FP16训练
    test_fp16_training()
    
    # 测试批次大小扩展
    test_batch_size_scaling()
    
    print("\n" + "="*60)
    print("Performance test completed!")
    print("="*60)
