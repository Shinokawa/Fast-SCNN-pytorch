"""BDD100K Drivable Area Segmentation Dataloader"""
import os
import random
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image, ImageOps, ImageFilter

__all__ = ['BDD100KSegmentation']


class BDD100KSegmentation(data.Dataset):
    """BDD100K Drivable Area Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to BDD100K folder. Default is './bdd100k'
    split: string
        'train', 'val' or 'test'
    subset: string
        '10k' or '100k' - which subset to use
    label_type: string
        'binary' or 'ternary' 
        - binary: 2 classes (background, drivable)
        - ternary: 3 classes (background, direct drivable, alternative drivable)
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset with sampling
    >>> trainset = BDD100KSegmentation(split='train', subset='100k', label_type='binary', 
    ...                                transform=input_transform, sample_ratio=0.1)
    >>> # Or limit to specific number of samples
    >>> trainset = BDD100KSegmentation(split='train', subset='100k', label_type='binary',
    ...                                transform=input_transform, max_samples=5000)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'bdd100k'
    # Default to binary classification, will be updated in __init__
    NUM_CLASS = 2

    def __init__(self, root='./bdd100k', split='train', subset='100k', label_type='binary',
                 mode=None, transform=None, base_size=520, crop_size=480, 
                 sample_ratio=1.0, max_samples=None, 
                 # 新增参数
                 multi_scale=False, keep_original_size=False, 
                 min_scale=0.8, max_scale=1.2, **kwargs):
        super(BDD100KSegmentation, self).__init__()
        self.root = root
        self.split = split
        self.subset = subset
        self.label_type = label_type
        self.mode = mode if mode is not None else split
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.sample_ratio = sample_ratio  # 采样比例 (0.0-1.0)
        self.max_samples = max_samples    # 最大样本数量
        
        # 新增的多尺度训练参数
        self.multi_scale = multi_scale          # 是否使用多尺度训练
        self.keep_original_size = keep_original_size  # 是否保持原始尺寸
        self.min_scale = min_scale              # 最小缩放比例
        self.max_scale = max_scale              # 最大缩放比例
        
        # Set number of classes based on label type
        if label_type == 'binary':
            self.NUM_CLASS = 2  # background, drivable
        elif label_type == 'ternary':
            self.NUM_CLASS = 3  # background, direct drivable, alternative drivable
        else:
            raise ValueError(f"label_type must be 'binary' or 'ternary', got {label_type}")
        
        # Get image and mask pairs based on split and subset
        self.images, self.mask_paths = self._get_bdd100k_pairs(self.root, self.split, self.subset)
        
        # Apply sampling if specified
        if self.sample_ratio < 1.0 or self.max_samples is not None:
            original_size = len(self.images)
            self.images, self.mask_paths = self._apply_sampling(self.images, self.mask_paths)
            print(f'Applied sampling: {original_size} -> {len(self.images)} samples')
        
        assert (len(self.images) == len(self.mask_paths))
        
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + self.root + "\n")
        
        print(f'Found {len(self.images)} images for {self.split} split using {self.subset} subset')
        print(f'Label type: {self.label_type} ({self.NUM_CLASS} classes)')

    def _get_bdd100k_pairs(self, root, split, subset):
        """Get image and mask pairs from the BDD100K dataset."""
        # Image directory
        img_dir = os.path.join(root, 'images', subset, split)
        # Label directory (using grayscale labels)
        label_dir = os.path.join(root, 'drivable_maps', 'labels', split)
        
        if not os.path.exists(img_dir):
            raise RuntimeError(f"Image directory not found: {img_dir}")
        if not os.path.exists(label_dir):
            raise RuntimeError(f"Label directory not found: {label_dir}")
        
        img_paths = []
        mask_paths = []
        
        # Get all image files
        img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        img_files.sort()
        
        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            
            # Convert image filename to label filename
            # Image: 0000f77c-6257be58.jpg -> Label: 0000f77c-6257be58_drivable_id.png
            base_name = os.path.splitext(img_file)[0]
            label_file = f"{base_name}_drivable_id.png"
            label_path = os.path.join(label_dir, label_file)
            
            if os.path.exists(label_path):
                img_paths.append(img_path)
                mask_paths.append(label_path)
            else:
                print(f'Warning: Label not found for {img_file}: {label_path}')
        
        print(f'Matched {len(img_paths)} image-label pairs out of {len(img_files)} images')
        return img_paths, mask_paths

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        
        mask = Image.open(self.mask_paths[index])
        
        # synchronized transform
        if self.mode == 'train':
            if self.keep_original_size:
                img, mask = self._original_size_transform(img, mask)
            elif self.multi_scale:
                img, mask = self._multi_scale_transform(img, mask)
            else:
                img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            if self.keep_original_size:
                img, mask = self._original_size_transform(img, mask)
            else:
                img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
            
        return img, mask

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        crop_size = self.crop_size
        # random scale (short edge)
        if self.multi_scale:
            short_size = random.randint(int(self.base_size * self.min_scale), int(self.base_size * self.max_scale))
        else:
            short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _original_size_transform(self, img, mask):
        """保持原始尺寸的变换 - 只做数据增强，不裁剪"""
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        # optional gaussian blur
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _multi_scale_transform(self, img, mask):
        """多尺度变换 - 随机缩放但不裁剪"""
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        
        # random scale
        scale_factor = random.uniform(self.min_scale, self.max_scale)
        w, h = img.size
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        
        img = img.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)
        
        # optional gaussian blur
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        """Transform mask for drivable area segmentation.
        
        For BDD100K drivable area labels:
        - Value 0: background -> class 0 (not drivable)
        - Value 1: direct drivable area -> class 1 (red in color labels, safe for driving)
        - Value 2: alternative drivable area -> class 0 (blue in color labels, avoid for smart car)
        
        For binary classification: only red areas (value 1) are drivable.
        For ternary classification: all three classes are preserved.
        """
        mask = np.array(mask)
        
        # Convert to grayscale if it's RGB
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        mask = mask.astype('int32')
        
        if self.label_type == 'binary':
            # Convert to binary: only value 1 (red/direct drivable) becomes class 1
            # Both 0 (background) and 2 (blue/alternative) become class 0
            binary_mask = np.where(mask == 1, 1, 0).astype('int32')
            return torch.LongTensor(binary_mask)
        else:  # ternary
            # Keep original values: 0, 1, 2
            # Clamp values to valid range in case there are other values
            mask = np.clip(mask, 0, 2).astype('int32')
            return torch.LongTensor(mask)

    def __len__(self):
        return len(self.images)

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0

    def _apply_sampling(self, images, mask_paths):
        """Apply sampling to reduce dataset size for experimentation."""
        total_samples = len(images)
        
        # Determine target sample count
        if self.max_samples is not None:
            target_samples = min(self.max_samples, total_samples)
        else:
            target_samples = int(total_samples * self.sample_ratio)
        
        if target_samples >= total_samples:
            return images, mask_paths
        
        # Random sampling with fixed seed for reproducibility
        import random
        random.seed(42)  # Fixed seed for reproducible sampling
        indices = random.sample(range(total_samples), target_samples)
        indices.sort()  # Keep original order
        
        sampled_images = [images[i] for i in indices]
        sampled_masks = [mask_paths[i] for i in indices]
        
        return sampled_images, sampled_masks

if __name__ == '__main__':
    # Test the dataset
    print("Testing BDD100K Dataset with 100K images...")
    
    # Test with sampling - use only 10% of data for quick testing
    print("\n=== Testing with sampling (10% of data) ===")
    dataset_sample = BDD100KSegmentation(split='train', subset='100k', label_type='binary', 
                                        sample_ratio=0.1)
    print(f'Sampled dataset size: {len(dataset_sample)}')
    if len(dataset_sample) > 0:
        img, mask = dataset_sample[0]
        print(f'Image shape: {img.shape}')
        print(f'Mask shape: {mask.shape}')
        print(f'Mask unique values: {torch.unique(mask)}')
    
    print("\n=== Testing with max samples limit ===")
    # Test with max samples limit
    dataset_limited = BDD100KSegmentation(split='train', subset='100k', label_type='binary',
                                         max_samples=1000)
    print(f'Limited dataset size: {len(dataset_limited)}')
    
    print("\n=== Testing ternary classification ===")
    # Test ternary classification
    dataset_ternary = BDD100KSegmentation(split='train', subset='100k', label_type='ternary',
                                         sample_ratio=0.05)  # Even smaller sample for ternary
    print(f'Ternary dataset size: {len(dataset_ternary)}')
    if len(dataset_ternary) > 0:
        img, mask = dataset_ternary[0]
        print(f'Ternary - Image shape: {img.shape}')
        print(f'Ternary - Mask shape: {mask.shape}')
        print(f'Ternary - Mask unique values: {torch.unique(mask)}')
    
    print("\n=== Full dataset info (no sampling) ===")
    # Show full dataset size without sampling
    dataset_full = BDD100KSegmentation(split='train', subset='100k', label_type='binary')
    print(f'Full dataset size: {len(dataset_full)}')
    
    print("\n=== Recommended configurations ===")
    print("For quick experimentation:")
    print("  - sample_ratio=0.1 (10% of data)")
    print("  - max_samples=5000 (fixed 5K samples)")
    print("For full training:")
    print("  - sample_ratio=1.0 or max_samples=None (use all data)")
