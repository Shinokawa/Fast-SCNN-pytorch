#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于BDD100K预训练模型对自定义数据集进行微调训练
支持原尺寸训练和多尺度训练
"""

import os
import argparse
import time
import datetime
import sys
import shutil

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from models.fast_scnn import get_fast_scnn
from data_loader.custom import CustomDataset
from utils.loss import MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss
from utils.lr_scheduler import LRScheduler
from utils.metric import SegmentationMetric
from training_visualizer import TrainingMonitor

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device

        # 数据集设置
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        
        # 训练集
        train_dataset = CustomDataset(
            root=args.data_folder,
            split='train',
            mode='train',
            base_size=args.base_size,
            crop_size=args.crop_size,
            original_size=args.original_size,
            multi_scale=args.multi_scale
        )
        
        # 验证集
        val_dataset = CustomDataset(
            root=args.data_folder,
            split='val', 
            mode='val',
            base_size=args.base_size,
            crop_size=args.crop_size,
            original_size=args.original_size,
            multi_scale=False  # 验证时不使用多尺度
        )
        
        self.train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                          shuffle=True, drop_last=True, **kwargs)
        self.val_loader = data.DataLoader(val_dataset, batch_size=1,
                                        shuffle=False, drop_last=False, **kwargs)
        
        self.num_classes = train_dataset.num_class
        print(f"自定义数据集类别数: {self.num_classes}")

        # 模型
        self.model = get_fast_scnn(dataset='custom', aux=args.aux).to(self.device)

        # 加载BDD100K预训练权重
        if args.pretrained_model:
            print(f"加载BDD100K预训练模型: {args.pretrained_model}")
            checkpoint = torch.load(args.pretrained_model, map_location=self.device)
            
            # 提取模型状态字典
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                pretrained_dict = checkpoint['model']
            else:
                pretrained_dict = checkpoint
            
            # 获取当前模型状态字典
            model_dict = self.model.state_dict()
            
            # 过滤掉不匹配的层（主要是分类头）
            filtered_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                    print(f"加载层: {k}")
                else:
                    print(f"跳过层: {k} (形状不匹配或不存在)")
            
            # 更新模型状态字典
            model_dict.update(filtered_dict)
            self.model.load_state_dict(model_dict)
            print(f"成功加载 {len(filtered_dict)}/{len(pretrained_dict)} 层权重")

        # 损失函数
        self.criterion = MixSoftmaxCrossEntropyLoss(args.aux, args.aux_weight, ignore_index=-1).to(self.device)

        # 优化器
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                       lr=args.lr,
                                       momentum=args.momentum,
                                       weight_decay=args.weight_decay)

        # 学习率调度器
        self.lr_scheduler = LRScheduler(mode='poly', base_lr=args.lr, nepochs=args.epochs,
                                       iters_per_epoch=len(self.train_loader), power=0.9)

        # 评估指标
        self.metric = SegmentationMetric(self.num_classes)
        
        # 训练监控
        experiment_name = f"custom_finetune_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.monitor = TrainingMonitor(save_dir=args.save_folder, experiment_name=experiment_name)

        # 最佳模型记录
        self.best_pred = 0.0

    def train(self):
        epochs = self.args.epochs
        log_per_iters = self.args.log_iter
        start_time = time.time()
        self.model.train()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_losses = []
            
            for iteration, (images, targets) in enumerate(self.train_loader):
                # 学习率调度
                cur_lr = self.lr_scheduler(epoch * len(self.train_loader) + iteration)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = cur_lr

                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                epoch_losses.append(loss.item())

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (iteration + 1) % log_per_iters == 0:
                    eta_seconds = ((time.time() - start_time) / ((epoch * len(self.train_loader)) + iteration + 1)) * \
                                (epochs * len(self.train_loader) - (epoch * len(self.train_loader)) - iteration - 1)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    print('Epoch: [{}/{}] Iter:[{}/{}] lr: {:.6f} Loss: {:.4f} ETA: {}'.format(
                        epoch + 1, epochs, iteration + 1, len(self.train_loader),
                        cur_lr, loss.item(), eta_string))

            # 记录epoch损失
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            
            epoch_time = time.time() - epoch_start_time
            print(f'Epoch [{epoch + 1}/{epochs}] 完成，平均损失: {avg_loss:.4f}, 用时: {epoch_time:.2f}s')

            # 验证
            if (epoch + 1) % self.args.val_epoch == 0 or epoch == epochs - 1:
                self.validation(epoch + 1)

            # 保存检查点
            if (epoch + 1) % (epochs // 5) == 0 or epoch == epochs - 1:
                save_checkpoint(self.model, self.args, is_best=False, epoch=epoch + 1)

        total_training_time = time.time() - start_time
        print(f"训练完成! 总用时: {total_training_time / 3600:.2f} 小时")
        save_checkpoint(self.model, self.args, is_best=False, epoch=epochs)

    def validation(self, epoch):
        self.metric.reset()
        self.model.eval()
        
        val_losses = []
        for i, (image, target) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = self.model(image)
                loss = self.criterion(outputs, target)
                val_losses.append(loss.item())

            pred = torch.argmax(outputs[0], 1)
            pred = pred.cpu().data.numpy()
            target = target.cpu().numpy()
            self.metric.update(pred, target)

        # 计算指标
        pixAcc, mIoU = self.metric.get()
        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f'Epoch [{epoch}] 验证结果: Loss: {avg_val_loss:.4f}, pixAcc: {pixAcc:.4f}, mIoU: {mIoU:.4f}')

        # 检查是否是最佳模型
        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            print(f'新的最佳模型! 综合得分: {new_pred:.4f}')
        else:
            is_best = False
        
        save_checkpoint(self.model, self.args, is_best, epoch)
        self.model.train()  # 切换回训练模式


def save_checkpoint(model, args, is_best=False, epoch=None):
    """保存模型检查点"""
    model_state_dict = model.state_dict()
    
    checkpoint = {
        'model': model_state_dict,
        'epoch': epoch,
        'args': args
    }
    
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    if epoch is not None:
        filename = f'checkpoint_epoch_{epoch}.pth'
    else:
        filename = 'checkpoint.pth'
    
    checkpoint_path = os.path.join(args.save_folder, filename)
    torch.save(checkpoint, checkpoint_path)
    print(f'模型已保存: {checkpoint_path}')
    
    if is_best:
        best_path = os.path.join(args.save_folder, 'best_model.pth')
        shutil.copyfile(checkpoint_path, best_path)
        print(f'最佳模型已保存: {best_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='基于BDD100K预训练模型微调自定义数据集')
    
    # 数据集参数
    parser.add_argument('--data-folder', type=str, default='data/custom',
                        help='自定义数据集路径')
    parser.add_argument('--pretrained-model', type=str, 
                        default='weights/bdd100k_fastscnn_best.pth',
                        help='BDD100K预训练模型路径')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='fast_scnn',
                        help='模型名称 (fast_scnn, etc.)')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='辅助损失')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='辅助损失权重')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='训练轮数 (default: 50)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='开始轮数 (default: 0)')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='批次大小 (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='学习率 (default: 1e-4，微调使用较小学习率)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='动量 (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='权重衰减 (default: 1e-4)')
    parser.add_argument('--warmup-iters', type=int, default=0,
                        help='预热迭代数')
    
    # 数据增强参数
    parser.add_argument('--base-size', type=int, default=1024,
                        help='基础尺寸')
    parser.add_argument('--crop-size', type=int, default=768,
                        help='裁剪尺寸')
    parser.add_argument('--original-size', action='store_true', default=False,
                        help='使用原始尺寸训练')
    parser.add_argument('--multi-scale', action='store_true', default=False,
                        help='多尺度训练')
    
    # 系统参数
    parser.add_argument('--workers', type=int, default=4, metavar='N',
                        help='数据加载线程数 (default: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda 或 cpu)')
    
    # 日志和保存参数
    parser.add_argument('--log-iter', type=int, default=10,
                        help='日志打印间隔')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='验证间隔轮数')
    parser.add_argument('--save-folder', type=str, default='weights/custom_finetune',
                        help='模型保存文件夹')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # 创建保存目录
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device
    
    # 检查数据集
    if not os.path.exists(args.data_folder):
        print(f"错误: 数据集文件夹不存在: {args.data_folder}")
        return
    
    # 检查预训练模型
    if args.pretrained_model and not os.path.exists(args.pretrained_model):
        print(f"警告: 预训练模型不存在: {args.pretrained_model}")
        print("将从头开始训练")
        args.pretrained_model = None
    
    print("=" * 50)
    print("基于BDD100K预训练模型微调自定义数据集")
    print("=" * 50)
    print(f"数据集路径: {args.data_folder}")
    print(f"预训练模型: {args.pretrained_model}")
    print(f"模型: {args.model}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    print(f"原尺寸训练: {args.original_size}")
    print(f"多尺度训练: {args.multi_scale}")
    print(f"保存路径: {args.save_folder}")
    print(f"设备: {device}")
    print("=" * 50)
    
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
