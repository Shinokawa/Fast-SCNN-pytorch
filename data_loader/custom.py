import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    """
    自定义数据集加载器，支持二分类可驾驶区域分割
    - 兼容BDD100K的标签格式（0: 不可驾驶, 1: 可驾驶）
    - 支持训练/验证自动分割
    - 支持原尺寸和多尺度训练
    """
    
    NUM_CLASS = 2  # 类属性，用于模型创建
    
    def __init__(self, root='data/custom', split='train', mode='train', transform=None, 
                 base_size=2048, crop_size=1024, train_split=0.9,  # 改为0.9，验证集0.1
                 keep_original_size=False, multi_scale=False, 
                 sample_ratio=1.0, max_samples=None, **kwargs):
        self.root = root
        self.split = split
        self.mode = mode
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.keep_original_size = keep_original_size
        self.multi_scale = multi_scale
        self.sample_ratio = sample_ratio
        self.max_samples = max_samples
        
        self.images_root = os.path.join(self.root, 'images')
        self.masks_root = os.path.join(self.root, 'masks')
        
        # 获取所有文件名
        all_filenames = []
        for filename in os.listdir(self.images_root):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                basename = os.path.splitext(filename)[0]
                mask_path = os.path.join(self.masks_root, basename + '.png')
                if os.path.exists(mask_path):
                    all_filenames.append(basename)
        
        if not all_filenames:
            raise FileNotFoundError(f"No matching image-mask pairs found in {self.root}")
        
        # 训练/验证分割
        if len(all_filenames) == 1:
            # 如果只有一个文件，用作训练和验证
            self.filenames = all_filenames
        else:
            train_files, val_files = train_test_split(
                all_filenames, train_size=train_split, random_state=42
            )
            
            if split == 'train':
                self.filenames = train_files
            elif split == 'val':
                self.filenames = val_files
            else:
                self.filenames = all_filenames
        
        print(f"自定义数据集 ({split}): {len(self.filenames)} 个样本")
        
        # 多尺度训练的尺度
        if multi_scale:
            self.scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    def __getitem__(self, index):
        filename = self.filenames[index]
        
        # 支持多种图片格式
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_path = os.path.join(self.images_root, filename + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            raise FileNotFoundError(f"Image not found for {filename}")
            
        mask_path = os.path.join(self.masks_root, filename + '.png')
        
        image = Image.open(image_path).convert('RGB')
        
        if self.mode == 'test':
            if self.transform is not None:
                image = self.transform(image)
            return image, os.path.basename(image_path)

        mask = Image.open(mask_path)
        
        # 转换为二分类标签 (0: 不可驾驶, 1: 可驾驶)
        mask_array = np.array(mask)
        if len(mask_array.shape) == 3:
            # 彩色mask，转换为灰度
            mask_array = mask_array[:, :, 0]  # 取第一个通道
        
        # 黑白mask: 白色(255)为可驾驶区域(1)，黑色(0)为不可驾驶区域(0)
        # 使用阈值128来处理可能的抗锯齿边缘
        binary_mask = (mask_array > 128).astype(np.uint8)
        mask = Image.fromarray(binary_mask)

        if self.mode == 'train':
            image, mask = self._sync_transform(image, mask)
        elif self.mode == 'val':
            image, mask = self._val_sync_transform(image, mask)
        else:
            raise NotImplementedError

        # 注意：_sync_transform 和 _val_sync_transform 已经处理了转换
        # 不需要再应用 self.transform，因为已经转换为 tensor
            
        return image, mask

    def __len__(self):
        return len(self.filenames)

    def _sync_transform(self, img, mask):
        """训练时的数据增强"""
        if not self.keep_original_size:
            # 多尺度训练
            if self.multi_scale:
                scale = random.choice(self.scales)
                w, h = img.size
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.BILINEAR)
                mask = mask.resize((new_w, new_h), Image.NEAREST)
            
            # 随机裁剪
            w, h = img.size
            if w < self.crop_size or h < self.crop_size:
                # 如果图像小于裁剪尺寸，进行填充
                new_w = max(w, self.crop_size)
                new_h = max(h, self.crop_size)
                img = img.resize((new_w, new_h), Image.BILINEAR)
                mask = mask.resize((new_w, new_h), Image.NEAREST)
            
            # 随机裁剪
            w, h = img.size
            x1 = random.randint(0, w - self.crop_size)
            y1 = random.randint(0, h - self.crop_size)
            img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        else:
            # 原尺寸训练，但需要统一到固定尺寸以便批处理
            img = img.resize((self.base_size, self.base_size), Image.BILINEAR)
            mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)
        
        # 随机水平翻转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 转换为tensor
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        
        mask = torch.from_numpy(np.array(mask)).long()
        
        return img, mask

    def _val_sync_transform(self, img, mask):
        """验证时的变换"""
        if not self.keep_original_size:
            # 调整到基础大小
            img = img.resize((self.base_size, self.base_size), Image.BILINEAR)
            mask = mask.resize((self.base_size, self.base_size), Image.NEAREST)
        
        # 转换为tensor
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        
        mask = torch.from_numpy(np.array(mask)).long()
        
        return img, mask

    @property
    def num_class(self):
        return 2  # 二分类：不可驾驶(0), 可驾驶(1)

    @property
    def pred_offset(self):
        return 0
