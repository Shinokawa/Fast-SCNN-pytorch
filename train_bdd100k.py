#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training script for BDD100K Drivable Area Segmentation"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler

from torchvision import transforms
from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from utils.loss import MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss, DiceLoss, MixDiceLoss, FocalDiceLoss
from utils.lr_scheduler import LRScheduler
from utils.metric import SegmentationMetric


def parse_args():
    """Training Options for BDD100K Drivable Area Segmentation"""
    parser = argparse.ArgumentParser(description='Fast-SCNN for BDD100K Drivable Area Segmentation')
    
    # model and dataset
    parser.add_argument('--model', type=str, default='fast_scnn',
                        help='model name (default: fast_scnn)')
    parser.add_argument('--dataset', type=str, default='bdd100k',
                        help='dataset name (default: bdd100k)')
    parser.add_argument('--subset', type=str, default='100k', choices=['10k', '100k'],
                        help='BDD100K subset to use (default: 100k)')
    parser.add_argument('--label-type', type=str, default='binary', choices=['binary', 'ternary'],
                        help='label type: binary (2 classes), ternary (3 classes)')
    parser.add_argument('--sample-ratio', type=float, default=0.1,
                        help='sampling ratio for quick experimentation (default: 0.1 means 10 percent)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='maximum number of samples to use (overrides sample-ratio)')
    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=768,
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='train',
                        help='dataset train split (default: train)')
    
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
                        help='number of epochs to train (default: 80)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=8,
                        metavar='N', help='input batch size for training (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    
    # loss function options
    parser.add_argument('--loss-type', type=str, default='dice', choices=['ce', 'dice', 'focal_dice'],
                        help='loss function type: ce (cross entropy), dice, focal_dice (default: dice)')
    
    # mixed precision training
    parser.add_argument('--use-fp16', action='store_true', default=True,
                        help='use mixed precision training (default: True)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers for data loading (default: 4)')
    
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-folder', default='./weights',
                        help='Directory for saving checkpoint models')
    
    # evaluation only
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--val-interval', type=int, default=5,
                        help='validation interval (default: 5)')
    parser.add_argument('--print-interval', type=int, default=50,
                        help='print interval for training logs (default: 50)')
    
    # experiment mode
    parser.add_argument('--experiment', action='store_true', default=False,
                        help='quick experiment mode with reduced data')
    
    # the parser
    args = parser.parse_args()
    
    # Experiment mode adjustments
    if args.experiment:
        print("🧪 Running in experiment mode - using reduced settings")
        args.sample_ratio = 0.05  # Use only 5% of data
        args.epochs = 20
        args.batch_size = 4
        args.val_interval = 2
        args.print_interval = 10
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    args.device = device
    print(args)
    return args


class BDD100KTrainer(object):
    def __init__(self, args):
        self.args = args
        
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        
        # dataset and dataloader with sampling
        data_kwargs = {
            'transform': input_transform, 
            'base_size': args.base_size, 
            'crop_size': args.crop_size,
            'subset': args.subset,
            'label_type': args.label_type,
            'sample_ratio': args.sample_ratio,
            'max_samples': args.max_samples
        }
        
        train_dataset = get_segmentation_dataset(args.dataset, split=args.train_split, mode='train', **data_kwargs)
        
        # For validation, use smaller sample to speed up
        val_data_kwargs = data_kwargs.copy()
        val_data_kwargs['sample_ratio'] = min(0.2, args.sample_ratio * 2)  # Use at most 20% for val
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='val', **val_data_kwargs)
        
        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=args.num_workers,
                                            pin_memory=True)
        
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=args.num_workers,
                                          pin_memory=True)

        # create network
        self.model = get_fast_scnn(dataset=args.dataset, aux=args.aux)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
        self.model.to(args.device)

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        if args.loss_type == 'dice':
            self.criterion = MixDiceLoss(aux=args.aux, aux_weight=args.aux_weight).to(args.device)
            print("Using Dice Loss for drivable area segmentation")
        elif args.loss_type == 'focal_dice':
            self.criterion = FocalDiceLoss().to(args.device)
            print("Using Focal + Dice Loss for drivable area segmentation")
        else:  # cross entropy
            self.criterion = MixSoftmaxCrossEntropyOHEMLoss(aux=args.aux, aux_weight=args.aux_weight,
                                                            ignore_index=-1).to(args.device)
            print("Using Cross Entropy Loss")

        # optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=args.lr,
                                          weight_decay=args.weight_decay)

        # mixed precision scaler
        self.scaler = GradScaler() if args.use_fp16 else None
        if args.use_fp16:
            print("Using FP16 mixed precision training")

        # lr scheduling
        self.lr_scheduler = LRScheduler(mode='poly', base_lr=args.lr, nepochs=args.epochs,
                                        iters_per_epoch=len(self.train_loader), power=0.9)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)

        self.best_pred = 0.0
        
        # Performance tracking
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_miou_history = []
        self.val_pixacc_history = []
        
        self._print_config(train_dataset, val_dataset)

    def _print_config(self, train_dataset, val_dataset):
        print(f"\n{'='*60}")
        print(f"🚗 BDD100K Drivable Area Segmentation Training Configuration")
        print(f"{'='*60}")
        print(f"Dataset: {self.args.dataset} ({self.args.subset})")
        print(f"Label Type: {self.args.label_type} ({train_dataset.num_class} classes)")
        print(f"Model: {self.args.model}")
        print(f"Loss Type: {self.args.loss_type}")
        print(f"Optimizer: AdamW")
        print(f"Batch Size: {self.args.batch_size}")
        print(f"Learning Rate: {self.args.lr}")
        print(f"Epochs: {self.args.epochs}")
        print(f"Mixed Precision: {self.args.use_fp16}")
        print(f"Auxiliary Loss: {self.args.aux}")
        print(f"Sample Ratio: {self.args.sample_ratio*100:.1f} percent")
        if self.args.max_samples:
            print(f"Max Samples: {self.args.max_samples}")
        print(f"Validation: {'Disabled' if self.args.no_val else f'Every {self.args.val_interval} epochs'}")
        print(f"Train Dataset Size: {len(train_dataset):,}")
        print(f"Val Dataset Size: {len(val_dataset):,}")
        print(f"Iterations per Epoch: {len(self.train_loader):,}")
        print(f"Total Iterations: {len(self.train_loader) * self.args.epochs:,}")
        print(f"Device: {self.args.device}")
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("="*60)

    def train(self):
        cur_iters = 0
        total_start_time = time.time()
        print(f"\n🚀 Starting BDD100K drivable area segmentation training...")
        
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_start_time = time.time()

            for i, (images, targets) in enumerate(self.train_loader):
                iter_start_time = time.time()
                
                cur_lr = self.lr_scheduler(cur_iters)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = cur_lr

                images = images.to(self.args.device, non_blocking=True)
                targets = targets.to(self.args.device, non_blocking=True)

                self.optimizer.zero_grad()
                
                if self.args.use_fp16:
                    # Mixed precision training
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard training
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()

                epoch_loss += loss.item()
                cur_iters += 1
                
                if cur_iters % self.args.print_interval == 0:
                    avg_loss = epoch_loss / (i + 1)
                    iter_time = time.time() - iter_start_time
                    eta = iter_time * (len(self.train_loader) * self.args.epochs - cur_iters)
                    print(f'Epoch: [{epoch+1}/{self.args.epochs}] '
                          f'Iter: [{cur_iters}/{len(self.train_loader) * self.args.epochs}] '
                          f'Loss: {avg_loss:.4f} '
                          f'LR: {cur_lr:.2e} '
                          f'Time: {iter_time:.2f}s '
                          f'ETA: {eta/3600:.1f}h')

            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            self.train_loss_history.append(avg_epoch_loss)
            
            print(f'✅ Epoch [{epoch+1}/{self.args.epochs}] completed in {epoch_time:.1f}s, '
                  f'Avg Loss: {avg_epoch_loss:.4f}')
            
            # validation
            if not self.args.no_val and (epoch + 1) % self.args.val_interval == 0:
                is_best = self.validation(epoch + 1)
                self.save_checkpoint(is_best, epoch + 1)

        total_time = time.time() - total_start_time
        print(f"\n🎉 Training completed!")
        print(f"Total training time: {total_time/3600:.2f} hours")
        print(f"Average time per epoch: {total_time/self.args.epochs:.2f} seconds")
        
        # Print training summary
        if self.train_loss_history:
            print(f"Final training loss: {self.train_loss_history[-1]:.4f}")
            print(f"Best training loss: {min(self.train_loss_history):.4f}")
        
        if self.val_miou_history:
            print(f"Best validation mIoU: {max(self.val_miou_history):.4f}")
            print(f"Best validation PixAcc: {max(self.val_pixacc_history):.4f}")

        self.save_checkpoint(False, self.args.epochs, final=True)

    def validation(self, epoch):
        print(f"\n🔍 Running validation for epoch {epoch}...")
        is_best = False
        self.metric.reset()
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for i, (image, target) in enumerate(self.val_loader):
                image = image.to(self.args.device)
                target = target.to(self.args.device)
                
                if self.args.use_fp16:
                    with autocast():
                        outputs = self.model(image)
                        loss = self.criterion(outputs, target)
                else:
                    outputs = self.model(image)
                    loss = self.criterion(outputs, target)
                
                val_loss += loss.item()
                
                # Get predictions
                if isinstance(outputs, tuple):
                    pred = outputs[0]  # main output
                else:
                    pred = outputs
                
                pred = torch.argmax(pred, dim=1)
                self.metric.update(pred.cpu().numpy(), target.cpu().numpy())
                
                if (i + 1) % 200 == 0:
                    print(f'   Validation [{i+1}/{len(self.val_loader)}]')

        avg_val_loss = val_loss / len(self.val_loader)
        pixAcc, mIoU = self.metric.get()
        
        print(f'📊 Validation Results:')
        print(f'   Loss: {avg_val_loss:.4f}')
        print(f'   PixAcc: {pixAcc:.4f}')
        print(f'   mIoU: {mIoU:.4f}')
        
        self.val_loss_history.append(avg_val_loss)
        self.val_miou_history.append(mIoU)
        self.val_pixacc_history.append(pixAcc)
        
        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            print(f'🌟 New best mIoU: {self.best_pred:.4f}')
        
        return is_best

    def save_checkpoint(self, is_best=False, epoch=None, final=False):
        """Save Checkpoint"""
        directory = os.path.expanduser(self.args.save_folder)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if final:
            filename = f'{self.args.model}_{self.args.dataset}_{self.args.label_type}_final.pth'
        else:
            filename = f'{self.args.model}_{self.args.dataset}_{self.args.label_type}.pth'
        
        save_path = os.path.join(directory, filename)
        
        # Save model state
        if isinstance(self.model, nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
            
        torch.save(model_state, save_path)
        print(f"   💾 Model saved to: {save_path}")
        
        if is_best:
            best_filename = f'{self.args.model}_{self.args.dataset}_{self.args.label_type}_best.pth'
            best_path = os.path.join(directory, best_filename)
            torch.save(model_state, best_path)
            print(f"   🏆 Best model saved to: {best_path}")


def main():
    args = parse_args()
    trainer = BDD100KTrainer(args)
    
    if args.eval:
        print("Evaluation mode not implemented yet")
    else:
        trainer.train()


if __name__ == '__main__':
    main()
