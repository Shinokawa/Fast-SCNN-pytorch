import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root, split='train', mode='train', transform=None, **kwargs):
        self.root = root
        self.split = split
        self.mode = mode
        self.transform = transform
        self.images_root = os.path.join(self.root, 'images')
        self.masks_root = os.path.join(self.root, 'masks')
        
        self.filenames = [os.path.splitext(filename)[0] for filename in os.listdir(self.images_root)]
        if not self.filenames:
            raise FileNotFoundError(f"No images found in {self.images_root}")

    def __getitem__(self, index):
        filename = self.filenames[index]
        
        image_path = os.path.join(self.images_root, filename + '.jpg')
        mask_path = os.path.join(self.masks_root, filename + '.png')
        
        image = Image.open(image_path).convert('RGB')
        
        if self.mode == 'test':
            if self.transform is not None:
                image = self.transform(image)
            return image, os.path.basename(image_path)

        mask = Image.open(mask_path)

        if self.mode == 'train':
            image, mask = self._sync_transform(image, mask)
        elif self.mode == 'val':
            image, mask = self._val_sync_transform(image, mask)
        else:
            raise NotImplementedError

        if self.transform is not None:
            image = self.transform(image)
            
        return image, mask

    def __len__(self):
        return len(self.filenames)

    def _sync_transform(self, img, mask):
        # Add your desired training-time transformations here
        # For now, we'll just resize
        return img, mask

    def _val_sync_transform(self, img, mask):
        # Add your desired validation-time transformations here
        return img, mask

    @property
    def num_class(self):
        return 2 # Assuming binary segmentation (lane vs. background)

    @property
    def pred_offset(self):
        return 0
