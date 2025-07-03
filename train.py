import os
import argparse
import time
import shutil
import datetime

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler

from torchvision import transforms
from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from utils.loss import MixSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyOHEMLoss, DiceLoss, MixDiceLoss, FocalDiceLoss
from utils.lr_scheduler import LRScheduler
from utils.metric import SegmentationMetric
from training_visualizer import TrainingMonitor


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Fast-SCNN on PyTorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='fast_scnn',
                        help='model name (default: fast_scnn)')
    parser.add_argument('--dataset', type=str, default='citys',
                        help='dataset name (default: citys)')
    # BDD100K specific arguments
    parser.add_argument('--subset', type=str, default='100k', choices=['10k', '100k'],
                        help='BDD100K subset to use (default: 100k)')
    parser.add_argument('--label-type', type=str, default='binary', choices=['binary', 'ternary'],
                        help='BDD100K label type: binary (2 classes), ternary (3 classes)')
    parser.add_argument('--sample-ratio', type=float, default=1.0,
                        help='sampling ratio for quick experimentation (default: 1.0 means 100 percent)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='maximum number of samples to use (overrides sample-ratio)')
    parser.add_argument('--keep-original-size', action='store_true', default=False,
                        help='keep original image size without cropping (for full scene training)')
    parser.add_argument('--multi-scale', action='store_true', default=False,
                        help='use multi-scale training without cropping')
    parser.add_argument('--min-scale', type=float, default=0.8,
                        help='minimum scale factor for multi-scale training')
    parser.add_argument('--max-scale', type=float, default=1.2,
                        help='maximum scale factor for multi-scale training')
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
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=16,
                        metavar='N', help='input batch size for training (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-2)')
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
    parser.add_argument('--val-interval', type=int, default=1,
                        help='validation interval (default: 1)')
    parser.add_argument('--print-interval', type=int, default=10,
                        help='print interval for training logs (default: 10)')
    # the parser
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    args.device = device
    print(args)
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 'crop_size': args.crop_size}
        
        # Add BDD100K specific parameters if using BDD100K dataset
        if args.dataset == 'bdd100k':
            data_kwargs.update({
                'subset': args.subset,
                'label_type': args.label_type,
                'sample_ratio': args.sample_ratio,
                'max_samples': args.max_samples,
                'keep_original_size': args.keep_original_size,
                'multi_scale': args.multi_scale,
                'min_scale': args.min_scale,
                'max_scale': args.max_scale
            })
        
        # Add custom dataset specific parameters
        elif args.dataset == 'custom':
            data_kwargs.update({
                'sample_ratio': args.sample_ratio,
                'max_samples': args.max_samples,
                'keep_original_size': args.keep_original_size,
                'multi_scale': args.multi_scale,
                'min_scale': args.min_scale,
                'max_scale': args.max_scale
            })
        
        train_dataset = get_segmentation_dataset(args.dataset, split=args.train_split, mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(args.dataset, split='val', mode='val', **data_kwargs)
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
        # For BDD100K, determine num_classes based on label_type
        if args.dataset == 'bdd100k':
            if args.label_type == 'binary':
                num_classes = 2
            elif args.label_type == 'ternary':
                num_classes = 3
            else:
                raise ValueError(f"Invalid label_type: {args.label_type}")
            
            # Create model directly with explicit num_classes
            from models.fast_scnn import FastSCNN
            self.model = FastSCNN(num_classes=num_classes, aux=args.aux)
        # For custom dataset, use binary classification (2 classes)
        elif args.dataset == 'custom':
            from models.fast_scnn import FastSCNN
            self.model = FastSCNN(num_classes=2, aux=args.aux)
            print(f"Created Fast-SCNN model for custom dataset with 2 classes (binary segmentation)")
        else:
            # Use original method for other datasets
            self.model = get_fast_scnn(dataset=args.dataset, aux=args.aux)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2])
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
            print("Using Dice Loss for binary lane segmentation")
        elif args.loss_type == 'focal_dice':
            self.criterion = FocalDiceLoss().to(args.device)
            print("Using Focal + Dice Loss for binary lane segmentation")
        else:  # cross entropy
            self.criterion = MixSoftmaxCrossEntropyOHEMLoss(aux=args.aux, aux_weight=args.aux_weight,
                                                            ignore_index=-1).to(args.device)
            print("Using Cross Entropy Loss")

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=args.lr,
                                         momentum=args.momentum,
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
        
        # 训练监控器
        experiment_name = f"bdd100k_{args.label_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.monitor = TrainingMonitor(save_dir='./logs', experiment_name=experiment_name)
        self.monitor.log_config(args)

        print(f"\n=== Training Configuration ===")
        print(f"Dataset: {args.dataset}")
        print(f"Model: {args.model}")
        print(f"Loss Type: {args.loss_type}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Learning Rate: {args.lr}")
        print(f"Epochs: {args.epochs}")
        print(f"Mixed Precision: {args.use_fp16}")
        print(f"Auxiliary Loss: {args.aux}")
        print(f"Validation: {'Disabled' if args.no_val else f'Every {args.val_interval} epochs'}")
        print(f"Train Dataset Size: {len(train_dataset)}")
        print(f"Val Dataset Size: {len(val_dataset)}")
        print(f"Iterations per Epoch: {len(self.train_loader)}")
        print(f"Total Iterations: {len(self.train_loader) * args.epochs}")
        print("=" * 30)

    def train(self):
        cur_iters = 0
        total_start_time = time.time()
        print(f"\n🚀 Starting training...")
        
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_start_time = time.time()
            data_time = 0.0
            batch_time = 0.0

            for i, (images, targets) in enumerate(self.train_loader):
                data_end_time = time.time()
                iter_start_time = time.time()
                
                cur_lr = self.lr_scheduler(cur_iters)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = cur_lr

                images = images.to(self.args.device, non_blocking=True)
                targets = targets.to(self.args.device, non_blocking=True)
                data_time += time.time() - data_end_time

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
                
                batch_time += time.time() - iter_start_time
                
                if cur_iters % self.args.print_interval == 0:
                    avg_loss = epoch_loss / (i + 1)
                    avg_batch_time = batch_time / self.args.print_interval
                    avg_data_time = data_time / self.args.print_interval
                    
                    # Calculate samples per second
                    samples_per_sec = self.args.batch_size / avg_batch_time
                    
                    print('Epoch: [%2d/%2d] Iter [%4d/%4d] || '
                          'Time: %.4fs (Data: %.4fs) || '
                          'Speed: %.1f samples/s || '
                          'LR: %.8f || Loss: %.4f || Avg Loss: %.4f' % (
                        epoch, self.args.epochs, i + 1, len(self.train_loader),
                        avg_batch_time, avg_data_time, samples_per_sec, 
                        cur_lr, loss.item(), avg_loss))
                    
                    # Reset timers
                    batch_time = 0.0
                    data_time = 0.0

            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            self.train_loss_history.append(avg_epoch_loss)
            
            # Calculate epoch statistics
            samples_per_epoch = len(self.train_loader) * self.args.batch_size
            epoch_samples_per_sec = samples_per_epoch / epoch_time
            
            print('\n📊 Epoch [%d/%d] Summary:' % (epoch, self.args.epochs))
            print('   Time: %.2fs | Avg Loss: %.4f | Speed: %.1f samples/s' % (
                epoch_time, avg_epoch_loss, epoch_samples_per_sec))

            # Run validation
            val_metrics = None
            if not self.args.no_val and (epoch + 1) % self.args.val_interval == 0:
                val_metrics = self.validation(epoch)
                self.val_loss_history.append(val_metrics['loss'])
                self.val_miou_history.append(val_metrics['mIoU'])
                self.val_pixacc_history.append(val_metrics['pixAcc'])
            else:
                # Save checkpoint even without validation
                save_checkpoint(self.model, self.args, is_best=False)
            
            # 记录到监控器
            self.monitor.log_epoch(
                epoch=epoch + 1,
                train_loss=avg_epoch_loss,
                val_loss=val_metrics['loss'] if val_metrics else None,
                val_miou=val_metrics['mIoU'] if val_metrics else None,
                val_pixacc=val_metrics['pixAcc'] if val_metrics else None,
                learning_rate=cur_lr,
                epoch_time=epoch_time
            )
            
            print("-" * 80)

        total_time = time.time() - total_start_time
        print(f"\n🎉 Training completed!")
        print(f"Total training time: {total_time/3600:.2f} hours")
        print(f"Average time per epoch: {total_time/self.args.epochs:.2f} seconds")
        
        # Print training summary
        if self.train_loss_history:
            print(f"\n📈 Training Summary:")
            print(f"   Initial Loss: {self.train_loss_history[0]:.4f}")
            print(f"   Final Loss: {self.train_loss_history[-1]:.4f}")
            print(f"   Best Loss: {min(self.train_loss_history):.4f}")
        
        if self.val_miou_history:
            print(f"\n🎯 Validation Summary:")
            print(f"   Best mIoU: {max(self.val_miou_history)*100:.2f}%")
            print(f"   Best PixAcc: {max(self.val_pixacc_history)*100:.2f}%")
            print(f"   Final mIoU: {self.val_miou_history[-1]*100:.2f}%")
            print(f"   Final PixAcc: {self.val_pixacc_history[-1]*100:.2f}%")

        save_checkpoint(self.model, self.args, is_best=False)
        
        # 生成训练可视化和报告
        print(f"\n📊 生成训练分析报告...")
        self.monitor.plot_training_curves(save_plot=True)
        self.monitor.generate_report()

    def validation(self, epoch):
        print(f"\n🔍 Running validation for epoch {epoch}...")
        is_best = False
        self.metric.reset()
        self.model.eval()
        val_loss = 0.0
        val_start_time = time.time()
        
        with torch.no_grad():
            for i, (image, target) in enumerate(self.val_loader):
                image = image.to(self.args.device, non_blocking=True)
                target = target.to(self.args.device, non_blocking=True)

                if self.args.use_fp16:
                    with autocast():
                        outputs = self.model(image)
                        loss = self.criterion(outputs, target)
                else:
                    outputs = self.model(image)
                    loss = self.criterion(outputs, target)
                
                val_loss += loss.item()
                
                pred = torch.argmax(outputs[0], 1)
                pred = pred.cpu().data.numpy()
                target_np = target.cpu().numpy()
                self.metric.update(pred, target_np)
                
                if i % 100 == 0 and i > 0:
                    pixAcc, mIoU = self.metric.get()
                    print('   Progress: [%4d/%4d] || PixAcc: %.3f%% || mIoU: %.3f%% || Loss: %.4f' % (
                        i + 1, len(self.val_loader), pixAcc * 100, mIoU * 100, loss.item()))

        val_time = time.time() - val_start_time
        pixAcc, mIoU = self.metric.get()
        avg_val_loss = val_loss / len(self.val_loader)
        
        print(f'\n📊 Validation Results (Epoch {epoch}):')
        print(f'   Time: {val_time:.2f}s')
        print(f'   Average Loss: {avg_val_loss:.4f}')
        print(f'   Pixel Accuracy: {pixAcc * 100:.3f}%')
        print(f'   Mean IoU: {mIoU * 100:.3f}%')
        
        # Check if this is the best model
        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            print(f'   🎉 New best model! Combined Score: {new_pred * 100:.3f}%')
        else:
            print(f'   Best Score: {self.best_pred * 100:.3f}%')
        
        # Calculate per-class IoU for detailed analysis
        if hasattr(self.metric, 'iou_per_class'):
            iou_per_class = self.metric.iou_per_class()
            print(f'   Per-class IoU:')
            class_names = ['Background', 'Lane']
            for i, (class_name, iou) in enumerate(zip(class_names, iou_per_class)):
                print(f'     {class_name}: {iou * 100:.3f}%')
        
        save_checkpoint(self.model, self.args, is_best)
        
        # Return metrics for tracking
        return {
            'loss': avg_val_loss,
            'pixAcc': pixAcc,
            'mIoU': mIoU,
            'combined_score': new_pred,
            'is_best': is_best
        }


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_folder)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}.pth'.format(args.model, args.dataset)
    save_path = os.path.join(directory, filename)
    torch.save(model.state_dict(), save_path)
    print(f"   Model saved to: {save_path}")
    
    if is_best:
        best_filename = '{}_{}_best_model.pth'.format(args.model, args.dataset)
        best_save_path = os.path.join(directory, best_filename)
        shutil.copyfile(save_path, best_save_path)
        print(f"   Best model saved to: {best_save_path}")


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    if args.eval:
        print('Evaluation model: ', args.resume)
        trainer.validation(args.start_epoch)
    else:
        print('Starting Epoch: %d, Total Epochs: %d' % (args.start_epoch, args.epochs))
        trainer.train()
