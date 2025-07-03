#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize BDD100K labels to verify binary classification processing"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from data_loader.bdd100k import BDD100KSegmentation

def visualize_label_processing(dataset, num_samples=6, save_path='bdd100k_visualization.png'):
    """Visualize original and processed labels to verify binary classification."""
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    fig.suptitle('BDD100K Binary Classification Verification\n'
                 'Column 1: Original Image | Column 2: Original Color Label | '
                 'Column 3: Original Grayscale Label | Column 4: Processed Binary Label\n'
                 'Red areas should become white (class 1), everything else black (class 0)', 
                 fontsize=14, y=0.98)
    
    for i in range(num_samples):
        if i >= len(dataset):
            break
            
        # Get paths
        img_path = dataset.images[i]
        mask_path = dataset.mask_paths[i]
        
        # Load original image and resize to a standard size for display
        img = Image.open(img_path).convert('RGB')
        display_size = (512, 384)  # Standard display size
        img_resized = img.resize(display_size, Image.BILINEAR)
        img_array = np.array(img_resized)
        
        # Load original grayscale label and resize
        mask_gray = Image.open(mask_path)
        mask_gray_resized = mask_gray.resize(display_size, Image.NEAREST)
        mask_gray_array = np.array(mask_gray_resized)
        
        # Apply the same transform as the dataset to show actual processing
        # We'll manually apply the binary transformation
        mask_binary_array = np.where(mask_gray_array == 1, 1, 0).astype('int32')
        
        # Try to load color label for reference
        color_label_path = mask_path.replace('labels', 'colormaps').replace('_drivable_id.png', '_drivable_color.png')
        if os.path.exists(color_label_path):
            color_label = Image.open(color_label_path).convert('RGB')
            color_label_resized = color_label.resize(display_size, Image.NEAREST)
            color_label_array = np.array(color_label_resized)
        else:
            # Create pseudo-color from grayscale for visualization
            color_label_array = np.zeros_like(img_array)
            # Background (0) -> black
            # Direct drivable (1) -> red
            # Alternative drivable (2) -> blue
            color_label_array[mask_gray_array == 1] = [255, 0, 0]  # Red
            color_label_array[mask_gray_array == 2] = [0, 0, 255]  # Blue
        
        # Plot original image
        axes[i, 0].imshow(img_array)
        axes[i, 0].set_title(f'Sample {i+1}: Original Image')
        axes[i, 0].axis('off')
        
        # Plot color label (or pseudo-color)
        axes[i, 1].imshow(color_label_array)
        axes[i, 1].set_title('Color Label\n(Red=Direct, Blue=Alternative)')
        axes[i, 1].axis('off')
        
        # Plot original grayscale label with color mapping
        gray_colored = np.zeros((*mask_gray_array.shape, 3), dtype=np.uint8)
        gray_colored[mask_gray_array == 0] = [0, 0, 0]      # Black for background
        gray_colored[mask_gray_array == 1] = [255, 0, 0]    # Red for direct drivable
        gray_colored[mask_gray_array == 2] = [0, 0, 255]    # Blue for alternative drivable
        
        axes[i, 2].imshow(gray_colored)
        axes[i, 2].set_title(f'Grayscale Label\n0={np.sum(mask_gray_array==0)}, '
                            f'1={np.sum(mask_gray_array==1)}, '
                            f'2={np.sum(mask_gray_array==2)}')
        axes[i, 2].axis('off')
        
        # Plot processed binary label
        binary_colored = np.zeros((*mask_binary_array.shape, 3), dtype=np.uint8)
        binary_colored[mask_binary_array == 0] = [0, 0, 0]      # Black for not drivable
        binary_colored[mask_binary_array == 1] = [255, 255, 255] # White for drivable
        
        axes[i, 3].imshow(binary_colored)
        axes[i, 3].set_title(f'Binary Result\n0={np.sum(mask_binary_array==0)}, '
                            f'1={np.sum(mask_binary_array==1)}')
        axes[i, 3].axis('off')
        
        print(f'Sample {i+1}:')
        print(f'  Original label values: {np.unique(mask_gray_array)}')
        print(f'  Processed label values: {np.unique(mask_binary_array)}')
        print(f'  Original counts - 0: {np.sum(mask_gray_array==0)}, 1: {np.sum(mask_gray_array==1)}, 2: {np.sum(mask_gray_array==2)}')
        print(f'  Binary counts - 0: {np.sum(mask_binary_array==0)}, 1: {np.sum(mask_binary_array==1)}')
        print(f'  Binary ratio: {np.sum(mask_binary_array==1) / mask_binary_array.size * 100:.1f}% drivable')
        print(f'  ✅ Verification: Red pixels (1) -> White pixels (1): {np.sum(mask_gray_array==1) == np.sum(mask_binary_array==1)}')
        print()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Visualization saved to: {save_path}')
    plt.show()

def analyze_dataset_statistics(dataset, num_samples=100):
    """Analyze dataset statistics."""
    print(f"\n=== Dataset Statistics (analyzing {min(num_samples, len(dataset))} samples) ===")
    
    total_pixels = 0
    total_drivable_pixels = 0
    class_counts = {0: 0, 1: 0, 2: 0}
    
    for i in range(min(num_samples, len(dataset))):
        # Load original grayscale label
        mask_path = dataset.mask_paths[i]
        mask_gray = np.array(Image.open(mask_path))
        
        # Count original classes
        for class_id in [0, 1, 2]:
            class_counts[class_id] += np.sum(mask_gray == class_id)
        
        # Get processed binary result
        _, processed_mask = dataset[i]
        processed_mask_array = processed_mask.numpy()
        
        total_pixels += processed_mask_array.size
        total_drivable_pixels += np.sum(processed_mask_array == 1)
    
    print(f"Original label distribution:")
    total_original = sum(class_counts.values())
    print(f"  Class 0 (background): {class_counts[0]:,} ({class_counts[0]/total_original*100:.1f}%)")
    print(f"  Class 1 (red/direct): {class_counts[1]:,} ({class_counts[1]/total_original*100:.1f}%)")
    print(f"  Class 2 (blue/alternative): {class_counts[2]:,} ({class_counts[2]/total_original*100:.1f}%)")
    
    print(f"\nBinary classification result:")
    print(f"  Class 0 (not drivable): {total_pixels - total_drivable_pixels:,} ({(total_pixels - total_drivable_pixels)/total_pixels*100:.1f}%)")
    print(f"  Class 1 (drivable): {total_drivable_pixels:,} ({total_drivable_pixels/total_pixels*100:.1f}%)")
    
    print(f"\nVerification:")
    expected_drivable = class_counts[1]
    actual_drivable = total_drivable_pixels
    print(f"  Expected drivable pixels (only red): {expected_drivable:,}")
    print(f"  Actual drivable pixels: {actual_drivable:,}")
    print(f"  Match: {'✅ YES' if expected_drivable == actual_drivable else '❌ NO'}")

def main():
    print("🔍 BDD100K Binary Classification Verification")
    print("=" * 50)
    
    # Create dataset with small sample for visualization
    print("Loading BDD100K dataset...")
    dataset = BDD100KSegmentation(
        root='./bdd100k',
        split='train',
        subset='100k',
        label_type='binary',
        sample_ratio=0.01,  # Use 1% for quick visualization
        mode='val'  # Use val mode to avoid random transforms
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    if len(dataset) == 0:
        print("❌ No samples found! Please check dataset path and structure.")
        return
    
    # Analyze statistics
    analyze_dataset_statistics(dataset, num_samples=50)
    
    # Create visualization
    print("\n📊 Creating visualization...")
    visualize_label_processing(dataset, num_samples=6, save_path='bdd100k_binary_verification.png')
    
    print("\n✅ Verification complete!")
    print("Check the generated image 'bdd100k_binary_verification.png'")
    print("In the binary result (column 4):")
    print("  - White areas = drivable (should match red areas from column 2)")
    print("  - Black areas = not drivable (background + blue areas)")

if __name__ == '__main__':
    main()
