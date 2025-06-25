"""TUSimple Lane Segmentation Dataloader"""
import os
import random
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image, ImageOps, ImageFilter

__all__ = ['TUSimpleSegmentation']


class TUSimpleSegmentation(data.Dataset):
    """TUSimple Lane Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to TUSimple folder. Default is './manideep1108/tusimple/versions/5/TUSimple'
    split: string
        'train', 'val' or 'test'
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
    >>> # Create Dataset
    >>> trainset = TUSimpleSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'tusimple'
    NUM_CLASS = 2  # 2 classes: background (0) and lane (1)

    def __init__(self, root='./manideep1108/tusimple/versions/5/TUSimple', split='train', 
                 mode=None, transform=None, base_size=520, crop_size=480, **kwargs):
        super(TUSimpleSegmentation, self).__init__()
        self.root = root
        self.split = split
        self.mode = mode if mode is not None else split
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size
        
        # Get image and mask pairs based on split
        self.images, self.mask_paths = self._get_tusimple_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + self.root + "\n")
        
        print(f'Found {len(self.images)} images for {self.split} split')

    def _get_tusimple_pairs(self, root, split):
        """Get image and mask pairs from the TUSimple dataset."""
        if split == 'train' or split == 'val':
            # For training, use train_val_gt.txt
            gt_file = os.path.join(root, 'train_set', 'seg_label', 'list', 'train_val_gt.txt')
            train_clips_root = os.path.join(root, 'train_set')
            seg_label_root = os.path.join(root, 'train_set', 'seg_label')
        else:  # test
            # For testing, use test_gt.txt
            gt_file = os.path.join(root, 'train_set', 'seg_label', 'list', 'test_gt.txt')
            train_clips_root = os.path.join(root, 'test_set')
            seg_label_root = os.path.join(root, 'train_set', 'seg_label')
        
        img_paths = []
        mask_paths = []
        
        if not os.path.exists(gt_file):
            raise RuntimeError(f"Ground truth file not found: {gt_file}")
        
        with open(gt_file, 'r') as f:
            lines = f.readlines()
        
        # If split is train/val, we need to split the training data
        if split == 'train':
            # Use first 80% for training
            lines = lines[:int(0.9 * len(lines))]
        elif split == 'val':
            # Use last 20% for validation
            lines = lines[int(0.9 * len(lines)):]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
                
            img_rel_path = parts[0]  # e.g., /clips/0530/1492626760788443246_0/20.jpg
            mask_rel_path = parts[1]  # e.g., /seg_label/0530/1492626760788443246_0/20.png
            
            # Remove leading slash and construct full paths
            img_rel_path = img_rel_path.lstrip('/')
            mask_rel_path = mask_rel_path.lstrip('/')
            
            img_path = os.path.join(train_clips_root, img_rel_path)
            mask_path = os.path.join(seg_label_root, mask_rel_path.replace('seg_label/', ''))
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                img_paths.append(img_path)
                mask_paths.append(mask_path)
            else:
                print(f'Warning: Missing file - img: {img_path}, mask: {mask_path}')
        
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
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
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

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        """Transform mask to binary segmentation.
        
        For TUSimple dataset:
        - Value 0: background -> class 0
        - Values 2,3,4,5: lane lines -> class 1
        """
        mask = np.array(mask)
        
        # Convert to grayscale if it's RGB (take R channel as they should be the same)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        mask = mask.astype('int32')
        
        # Convert to binary mask: 0 stays 0, 2-5 become 1
        binary_mask = np.where(mask > 0, 1, 0).astype('int32')
        
        return torch.LongTensor(binary_mask)

    def __len__(self):
        return len(self.images)

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0


if __name__ == '__main__':
    # Test the dataset
    dataset = TUSimpleSegmentation(split='train')
    print(f'Dataset size: {len(dataset)}')
    if len(dataset) > 0:
        img, mask = dataset[0]
        print(f'Image shape: {img.shape}')
        print(f'Mask shape: {mask.shape}')
        print(f'Mask unique values: {torch.unique(mask)}')
