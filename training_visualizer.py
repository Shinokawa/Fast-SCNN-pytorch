#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练过程可视化和分析工具
用于监控训练过程、分析性能、便于调参
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import torch
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TrainingMonitor:
    """训练过程监控器"""
    
    def __init__(self, save_dir='./logs', experiment_name=None):
        self.save_dir = save_dir
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        self.log_file = os.path.join(save_dir, f"{experiment_name}_training_log.json")
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练数据存储
        self.data = {
            'train_loss': [],
            'val_loss': [],
            'val_miou': [],
            'val_pixacc': [],
            'learning_rates': [],
            'epoch_times': [],
            'config': {},
            'best_metrics': {},
            'timestamps': []
        }
        
        print(f"📊 训练监控器已启动: {self.experiment_name}")
        print(f"📁 日志保存到: {self.log_file}")

    def log_config(self, args):
        """记录训练配置"""
        config_dict = vars(args) if hasattr(args, '__dict__') else args
        # 转换device等不可序列化的对象
        serializable_config = {}
        for k, v in config_dict.items():
            if isinstance(v, torch.device):
                serializable_config[k] = str(v)
            else:
                serializable_config[k] = v
        
        self.data['config'] = serializable_config
        self.save_log()

    def log_epoch(self, epoch, train_loss, val_loss=None, val_miou=None, 
                  val_pixacc=None, learning_rate=None, epoch_time=None):
        """记录每个epoch的数据"""
        self.data['train_loss'].append(train_loss)
        if val_loss is not None:
            self.data['val_loss'].append(val_loss)
        if val_miou is not None:
            self.data['val_miou'].append(val_miou)
        if val_pixacc is not None:
            self.data['val_pixacc'].append(val_pixacc)
        if learning_rate is not None:
            self.data['learning_rates'].append(learning_rate)
        if epoch_time is not None:
            self.data['epoch_times'].append(epoch_time)
        
        self.data['timestamps'].append(datetime.now().isoformat())
        
        # 更新最佳指标
        if val_miou is not None:
            if 'best_miou' not in self.data['best_metrics'] or val_miou > self.data['best_metrics']['best_miou']:
                self.data['best_metrics']['best_miou'] = val_miou
                self.data['best_metrics']['best_miou_epoch'] = epoch
        
        if val_pixacc is not None:
            if 'best_pixacc' not in self.data['best_metrics'] or val_pixacc > self.data['best_metrics']['best_pixacc']:
                self.data['best_metrics']['best_pixacc'] = val_pixacc
                self.data['best_metrics']['best_pixacc_epoch'] = epoch
        
        self.save_log()

    def save_log(self):
        """保存日志到文件"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def plot_training_curves(self, save_plot=True):
        """绘制训练曲线"""
        epochs = list(range(1, len(self.data['train_loss']) + 1))
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'训练过程分析 - {self.experiment_name}', fontsize=16, fontweight='bold')
        
        # 1. 损失曲线
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.data['train_loss'], 'b-', label='训练损失', linewidth=2)
        if self.data['val_loss']:
            val_epochs = list(range(1, len(self.data['val_loss']) + 1))
            ax1.plot(val_epochs, self.data['val_loss'], 'r-', label='验证损失', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('损失曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. mIoU曲线
        ax2 = axes[0, 1]
        if self.data['val_miou']:
            val_epochs = list(range(1, len(self.data['val_miou']) + 1))
            ax2.plot(val_epochs, [x*100 for x in self.data['val_miou']], 'g-', 
                    label='验证mIoU', linewidth=2, marker='o', markersize=4)
            if 'best_miou' in self.data['best_metrics']:
                best_epoch = self.data['best_metrics']['best_miou_epoch']
                best_miou = self.data['best_metrics']['best_miou'] * 100
                ax2.axhline(y=best_miou, color='g', linestyle='--', alpha=0.7, 
                           label=f'最佳mIoU: {best_miou:.2f}%')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mIoU (%)')
        ax2.set_title('Mean IoU曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 像素准确率曲线
        ax3 = axes[1, 0]
        if self.data['val_pixacc']:
            val_epochs = list(range(1, len(self.data['val_pixacc']) + 1))
            ax3.plot(val_epochs, [x*100 for x in self.data['val_pixacc']], 'm-', 
                    label='像素准确率', linewidth=2, marker='s', markersize=4)
            if 'best_pixacc' in self.data['best_metrics']:
                best_pixacc = self.data['best_metrics']['best_pixacc'] * 100
                ax3.axhline(y=best_pixacc, color='m', linestyle='--', alpha=0.7,
                           label=f'最佳PixAcc: {best_pixacc:.2f}%')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Pixel Accuracy (%)')
        ax3.set_title('像素准确率曲线')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 学习率曲线
        ax4 = axes[1, 1]
        if self.data['learning_rates']:
            lr_epochs = list(range(1, len(self.data['learning_rates']) + 1))
            ax4.semilogy(lr_epochs, self.data['learning_rates'], 'orange', 
                        label='学习率', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate (log scale)')
        ax4.set_title('学习率变化')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.save_dir, f"{self.experiment_name}_training_curves.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📈 训练曲线已保存: {plot_path}")
        
        plt.show()
        return fig

    def generate_report(self):
        """生成训练报告"""
        if not self.data['train_loss']:
            print("❌ 没有训练数据可分析")
            return
        
        report = []
        report.append("=" * 60)
        report.append(f"🚗 BDD100K可驾驶区域分割训练报告")
        report.append(f"📅 实验名称: {self.experiment_name}")
        report.append(f"📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        # 训练配置
        if self.data['config']:
            report.append("\n📋 训练配置:")
            config = self.data['config']
            report.append(f"   数据集: {config.get('dataset', 'N/A')}")
            report.append(f"   模型: {config.get('model', 'N/A')}")
            report.append(f"   标签类型: {config.get('label_type', 'N/A')}")
            report.append(f"   批次大小: {config.get('batch_size', 'N/A')}")
            report.append(f"   学习率: {config.get('lr', 'N/A')}")
            report.append(f"   训练轮数: {config.get('epochs', 'N/A')}")
            report.append(f"   损失函数: {config.get('loss_type', 'N/A')}")
            report.append(f"   数据采样: {config.get('sample_ratio', 'N/A')}")
            report.append(f"   保持原尺寸: {config.get('keep_original_size', False)}")
            report.append(f"   多尺度训练: {config.get('multi_scale', False)}")
        
        # 训练结果
        report.append(f"\n📊 训练结果:")
        report.append(f"   总训练轮数: {len(self.data['train_loss'])}")
        report.append(f"   初始训练损失: {self.data['train_loss'][0]:.4f}")
        report.append(f"   最终训练损失: {self.data['train_loss'][-1]:.4f}")
        report.append(f"   最低训练损失: {min(self.data['train_loss']):.4f}")
        
        if self.data['val_miou']:
            report.append(f"   最终验证mIoU: {self.data['val_miou'][-1]*100:.2f}%")
            report.append(f"   最佳验证mIoU: {max(self.data['val_miou'])*100:.2f}%")
        
        if self.data['val_pixacc']:
            report.append(f"   最终像素准确率: {self.data['val_pixacc'][-1]*100:.2f}%")
            report.append(f"   最佳像素准确率: {max(self.data['val_pixacc'])*100:.2f}%")
        
        # 训练时间分析
        if self.data['epoch_times']:
            total_time = sum(self.data['epoch_times'])
            avg_time = np.mean(self.data['epoch_times'])
            report.append(f"\n⏱️  训练时间分析:")
            report.append(f"   总训练时间: {total_time/3600:.2f} 小时")
            report.append(f"   平均每轮时间: {avg_time:.1f} 秒")
            report.append(f"   最快一轮: {min(self.data['epoch_times']):.1f} 秒")
            report.append(f"   最慢一轮: {max(self.data['epoch_times']):.1f} 秒")
        
        # 收敛分析
        if len(self.data['train_loss']) >= 5:
            recent_losses = self.data['train_loss'][-5:]
            loss_trend = (recent_losses[-1] - recent_losses[0]) / len(recent_losses)
            report.append(f"\n📈 收敛分析 (最近5轮):")
            report.append(f"   损失趋势: {loss_trend:.6f} 每轮")
            if abs(loss_trend) < 0.001:
                report.append("   ✅ 模型已收敛")
            elif loss_trend < 0:
                report.append("   📉 模型仍在改善")
            else:
                report.append("   ⚠️  模型可能过拟合")
        
        # 调参建议
        report.append(f"\n💡 调参建议:")
        if self.data['val_miou']:
            final_miou = self.data['val_miou'][-1] * 100
            if final_miou < 50:
                report.append("   📌 mIoU较低，建议:")
                report.append("      - 增加训练轮数")
                report.append("      - 调整学习率")
                report.append("      - 检查数据质量")
            elif final_miou < 70:
                report.append("   📌 mIoU中等，建议:")
                report.append("      - 微调学习率调度")
                report.append("      - 尝试不同的数据增强")
                report.append("      - 增加训练数据")
            else:
                report.append("   ✅ mIoU表现良好！")
                report.append("      - 可以尝试增加数据量进一步提升")
                report.append("      - 可以进行多尺度训练")
        
        report_text = "\n".join(report)
        print(report_text)
        
        # 保存报告
        report_path = os.path.join(self.save_dir, f"{self.experiment_name}_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n📄 详细报告已保存: {report_path}")
        
        return report_text

def load_and_analyze_training(log_file):
    """加载并分析已有的训练日志"""
    if not os.path.exists(log_file):
        print(f"❌ 日志文件不存在: {log_file}")
        return None
    
    with open(log_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建监控器实例
    experiment_name = os.path.splitext(os.path.basename(log_file))[0].replace('_training_log', '')
    monitor = TrainingMonitor(save_dir=os.path.dirname(log_file), experiment_name=experiment_name)
    monitor.data = data
    
    return monitor

if __name__ == '__main__':
    # 示例：分析现有的训练日志
    print("🔍 训练分析工具")
    print("请将此脚本集成到train.py中，或提供日志文件路径进行分析")
    
    # 搜索现有的日志文件
    log_dir = './logs'
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith('_training_log.json')]
        if log_files:
            print(f"\n找到 {len(log_files)} 个训练日志:")
            for i, log_file in enumerate(log_files, 1):
                print(f"  {i}. {log_file}")
            
            # 可以选择分析特定的日志文件
            # latest_log = os.path.join(log_dir, log_files[-1])
            # monitor = load_and_analyze_training(latest_log)
            # if monitor:
            #     monitor.plot_training_curves()
            #     monitor.generate_report()
    else:
        print("没有找到日志目录，请先进行训练")
