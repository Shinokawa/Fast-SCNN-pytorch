#!/usr/bin/env python3
"""
Data Augmentation Script: Horizontal Flip
This script creates horizontally flipped versions of images and masks to double the dataset size
"""
import os
import cv2
import numpy as np
from pathlib import Path
import argparse

def flip_image_and_mask(image_path, mask_path, output_image_path, output_mask_path):
    """
    Horizontally flip both image and mask
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not read image {image_path}")
        return False
    
    # Read mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Could not read mask {mask_path}")
        return False
    
    # Horizontal flip
    flipped_image = cv2.flip(image, 1)  # 1 for horizontal flip
    flipped_mask = cv2.flip(mask, 1)
    
    # Save flipped image and mask
    cv2.imwrite(str(output_image_path), flipped_image)
    cv2.imwrite(str(output_mask_path), flipped_mask)
    
    return True

def augment_dataset(data_dir="data/custom", output_suffix="_flipped"):
    """
    Create horizontally flipped versions of all images and masks
    """
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    masks_dir = data_path / "masks"
    
    if not images_dir.exists() or not masks_dir.exists():
        print(f"Error: {images_dir} or {masks_dir} does not exist!")
        return False
    
    # Get all image files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    image_files.sort()
    
    print(f"Found {len(image_files)} images to augment")
    
    success_count = 0
    
    for image_file in image_files:
        # Get corresponding mask file
        mask_file = masks_dir / (image_file.stem + ".png")
        
        if not mask_file.exists():
            print(f"Warning: Mask not found for {image_file.name}")
            continue
        
        # Create output file names
        output_image_file = images_dir / (image_file.stem + output_suffix + image_file.suffix)
        output_mask_file = masks_dir / (mask_file.stem + output_suffix + mask_file.suffix)
        
        # Skip if output files already exist
        if output_image_file.exists() and output_mask_file.exists():
            print(f"Skipping {image_file.name} (flipped version already exists)")
            continue
        
        # Flip and save
        if flip_image_and_mask(image_file, mask_file, output_image_file, output_mask_file):
            success_count += 1
            print(f"✓ Created flipped version of {image_file.name}")
        else:
            print(f"✗ Failed to create flipped version of {image_file.name}")
    
    print(f"\n🎉 Data augmentation completed!")
    print(f"Successfully created {success_count} flipped image-mask pairs")
    
    # Count total files after augmentation
    total_images = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    total_masks = len(list(masks_dir.glob("*.png")))
    
    print(f"Total images after augmentation: {total_images}")
    print(f"Total masks after augmentation: {total_masks}")
    
    return True

def verify_flipped_data(data_dir="data/custom", output_suffix="_flipped"):
    """
    Verify that flipped images and masks are created correctly
    """
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    masks_dir = data_path / "masks"
    
    print(f"\n🔍 Verifying flipped data...")
    
    # Find flipped files
    flipped_images = list(images_dir.glob(f"*{output_suffix}.*"))
    flipped_masks = list(masks_dir.glob(f"*{output_suffix}.*"))
    
    print(f"Found {len(flipped_images)} flipped images")
    print(f"Found {len(flipped_masks)} flipped masks")
    
    # Check a few random pairs
    import random
    if flipped_images:
        sample_count = min(3, len(flipped_images))
        sample_images = random.sample(flipped_images, sample_count)
        
        for img_file in sample_images:
            mask_file = masks_dir / (img_file.stem + ".png")
            
            if mask_file.exists():
                # Read and check dimensions
                img = cv2.imread(str(img_file))
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                
                if img is not None and mask is not None:
                    print(f"✓ {img_file.name}: Image {img.shape}, Mask {mask.shape}")
                    
                    # Check mask values
                    unique_values = np.unique(mask)
                    print(f"  Mask unique values: {unique_values}")
                else:
                    print(f"✗ {img_file.name}: Failed to read files")
            else:
                print(f"✗ {img_file.name}: Corresponding mask not found")

def main():
    parser = argparse.ArgumentParser(description='Augment custom dataset with horizontal flips')
    parser.add_argument('--data-dir', default='data/custom', help='Path to custom dataset directory')
    parser.add_argument('--suffix', default='_flipped', help='Suffix for flipped files')
    parser.add_argument('--verify', action='store_true', help='Verify flipped data after creation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Data Augmentation: Horizontal Flip")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output suffix: {args.suffix}")
    print("=" * 60)
    
    # Perform augmentation
    success = augment_dataset(args.data_dir, args.suffix)
    
    if success and args.verify:
        verify_flipped_data(args.data_dir, args.suffix)
    
    if success:
        print("\n✅ Data augmentation completed successfully!")
        print("You can now train with the augmented dataset using:")
        print("python train.py --dataset custom --epochs 100 --batch-size 4 --lr 0.01")
    else:
        print("\n❌ Data augmentation failed!")

if __name__ == "__main__":
    main()
