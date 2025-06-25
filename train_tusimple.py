#!/usr/bin/env python3
"""
TUSimple Lane Segmentation Training Script
This script provides optimized training parameters for TUSimple lane segmentation dataset.
"""
import os
import argparse
import subprocess
import sys

def train_tusimple():
    """Train Fast-SCNN on TUSimple dataset with optimized parameters."""
    
    # Optimized training parameters for TUSimple with FP16 and Dice Loss
    cmd = [
        "python", "train.py",
        "--dataset", "tusimple",
        "--base-size", "512",
        "--crop-size", "480", 
        "--batch-size", "16",  # Increased batch size for better GPU utilization
        "--epochs", "100",
        "--lr", "0.01",
        "--aux",  # Use auxiliary loss
        "--aux-weight", "0.4",
        "--loss-type", "dice",  # Use Dice loss for better binary segmentation
        "--use-fp16",  # Enable mixed precision training
        "--num-workers", "4",  # Parallel data loading
        "--val-interval", "5",  # Validate every 5 epochs
        "--print-interval", "20"  # Print every 20 iterations
    ]
    
    print("Starting TUSimple Lane Segmentation Training with Optimizations...")
    print("Optimizations enabled:")
    print("  ✓ Dice Loss for binary segmentation")
    print("  ✓ FP16 Mixed Precision Training")
    print("  ✓ Batch Size: 16")
    print("  ✓ Parallel Data Loading (4 workers)")
    print("  ✓ Validation every 5 epochs")
    print("  ✓ Detailed performance monitoring")
    print("Command:", " ".join(cmd))
    print("-" * 60)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='TUSimple Lane Segmentation Training')
    parser.add_argument('--custom', action='store_true', 
                       help='Use custom parameters (will prompt for input)')
    
    args = parser.parse_args()
    
    if args.custom:
        print("Custom training parameters not implemented yet.")
        print("Please use: python train.py --dataset tusimple [other options]")
        sys.exit(1)
    else:
        train_tusimple()

if __name__ == '__main__':
    # Check if we're in the right directory
    if not os.path.exists('train.py'):
        print("Error: train.py not found. Please run this script from the Fast-SCNN-pytorch directory.")
        sys.exit(1)
    
    # Check if TUSimple dataset is available
    if not os.path.exists('./manideep1108/tusimple/versions/5/TUSimple/train_set/seg_label/list/train_val_gt.txt'):
        print("Error: TUSimple dataset not found. Please check your data directory.")
        sys.exit(1)
    
    main()
