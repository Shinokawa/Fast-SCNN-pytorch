#!/usr/bin/env python3
"""
Train Fast-SCNN from scratch on custom dataset
This script trains the model from scratch (no pretrained weights) to achieve overfitting on small custom dataset
"""
import os
import sys
import time
import subprocess

def run_training():
    """Run training from scratch on custom dataset"""
    
    # Ensure output directory exists
    os.makedirs("weights/custom_scratch", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Training parameters optimized for overfitting on small dataset
    cmd = [
        sys.executable, "train.py",
        "--dataset", "custom",
        "--epochs", "200",  # More epochs for overfitting
        "--batch-size", "4",  # Smaller batch size for better gradient updates
        "--lr", "0.01",  # Higher learning rate for faster convergence
        "--momentum", "0.9",
        "--weight-decay", "1e-5",  # Reduced weight decay to allow overfitting
        "--loss-type", "dice",  # Dice loss for binary segmentation
        "--aux",  # Use auxiliary loss
        "--aux-weight", "0.4",
        "--use-fp16",  # Mixed precision training
        "--num-workers", "2",
        "--save-folder", "./weights/custom_scratch",
        "--val-interval", "5",  # Validate every 5 epochs
        "--print-interval", "5",  # Print every 5 batches
        "--keep-original-size",  # Keep original size for better performance
        "--base-size", "1024",
        "--crop-size", "512"
    ]
    
    print("=" * 80)
    print("Training Fast-SCNN from scratch on custom dataset")
    print("Goal: Achieve overfitting on small custom dataset for verification")
    print("=" * 80)
    print("Training command:")
    print(" ".join(cmd))
    print("=" * 80)
    
    # Save command to log file
    log_file = "logs/custom_scratch_training.log"
    with open(log_file, "w") as f:
        f.write("Training command:\n")
        f.write(" ".join(cmd) + "\n")
        f.write("=" * 80 + "\n")
    
    # Run training
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=False, text=True, check=True)
        end_time = time.time()
        
        training_time = end_time - start_time
        print(f"\nTraining completed successfully!")
        print(f"Total training time: {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
        
        # Log completion
        with open(log_file, "a") as f:
            f.write(f"\nTraining completed successfully!\n")
            f.write(f"Total training time: {training_time:.2f} seconds\n")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Training failed with return code: {e.returncode}")
        with open(log_file, "a") as f:
            f.write(f"\nTraining failed with return code: {e.returncode}\n")
        return False
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        with open(log_file, "a") as f:
            f.write(f"\nTraining interrupted by user\n")
        return False

if __name__ == "__main__":
    success = run_training()
    if success:
        print("\n" + "=" * 80)
        print("Training completed! Check weights/custom_scratch/ for model checkpoints")
        print("Next steps:")
        print("1. Run inference to test the trained model")
        print("2. Visualize training results and segmentation quality")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("Training failed! Check logs/custom_scratch_training.log for details")
        print("=" * 80)
