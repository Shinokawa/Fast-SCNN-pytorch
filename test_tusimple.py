"""Test script for TUSimple dataset loader"""
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn

def check_paths_and_save_samples(dataset, split_name, num_samples=3):
    """Check actual file paths and save mask samples"""
    print(f"\n--- Checking {split_name} dataset paths and samples ---")
    
    # Create output directory for samples
    output_dir = f"./test_samples_{split_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(min(num_samples, len(dataset))):
        # Get raw paths from dataset
        img_path = dataset.images[i]
        mask_path = dataset.mask_paths[i]
        
        print(f"\nSample {i+1}:")
        print(f"  Image path: {img_path}")
        print(f"  Mask path:  {mask_path}")
        print(f"  Image exists: {os.path.exists(img_path)}")
        print(f"  Mask exists:  {os.path.exists(mask_path)}")
        
        if os.path.exists(img_path) and os.path.exists(mask_path):
            # Load original mask to check values
            original_mask = Image.open(mask_path)
            original_array = np.array(original_mask)
            
            print(f"  Original mask shape: {original_array.shape}")
            print(f"  Original mask mode: {original_mask.mode}")
            print(f"  Original unique values: {np.unique(original_array)}")
            
            # Get processed data from dataset
            if split_name == 'test':
                # Test mode returns (img_tensor, filename) instead of (img_tensor, mask_tensor)
                img_tensor, filename = dataset[i]
                print(f"  Test mode - filename: {filename}")
                
                # For test set, we need to load the mask manually
                mask_tensor = dataset._mask_transform(original_mask)
                
                print(f"  Manually loaded mask shape: {mask_tensor.shape}")
                print(f"  Manually loaded mask unique values: {torch.unique(mask_tensor)}")
            else:
                img_tensor, mask_tensor = dataset[i]
            
            # Convert tensors back to images for saving
            # Denormalize image tensor
            img_array = img_tensor.permute(1, 2, 0).numpy()
            img_array = img_array * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
            
            # Convert mask tensor to image
            mask_array = mask_tensor.numpy().astype(np.uint8) * 255  # 0->0, 1->255
            
            print(f"  Processed image shape: {img_array.shape}")
            print(f"  Processed mask shape: {mask_array.shape}")
            print(f"  Processed mask unique values: {np.unique(mask_array)}")
            
            # Save original images
            Image.fromarray(img_array).save(os.path.join(output_dir, f"sample_{i+1}_image_processed.jpg"))
            original_mask.save(os.path.join(output_dir, f"sample_{i+1}_mask_original.png"))
            Image.fromarray(mask_array).save(os.path.join(output_dir, f"sample_{i+1}_mask_processed.png"))
            
            # Copy original image
            original_img = Image.open(img_path)
            original_img.save(os.path.join(output_dir, f"sample_{i+1}_image_original.jpg"))
            
            print(f"  Saved samples to: {output_dir}")
        else:
            print(f"  ⚠️  Missing files!")
    
    print(f"\nSample images saved in: {output_dir}")

def test_tusimple_dataset():
    """Test TUSimple dataset loading"""
    print("Testing TUSimple dataset loader...")
    
    # Define transforms
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    
    # Test dataset creation
    try:
        # Create train dataset
        train_dataset = get_segmentation_dataset('tusimple', 
                                                split='train', 
                                                mode='train',
                                                transform=input_transform,
                                                base_size=512,
                                                crop_size=480)
        print(f"✓ Train dataset created successfully with {len(train_dataset)} samples")
        
        # Check train dataset paths and save samples
        check_paths_and_save_samples(train_dataset, "train", 3)
        
        # Create validation dataset
        val_dataset = get_segmentation_dataset('tusimple', 
                                              split='val', 
                                              mode='val',
                                              transform=input_transform,
                                              base_size=512,
                                              crop_size=480)
        print(f"✓ Validation dataset created successfully with {len(val_dataset)} samples")
        
        # Check validation dataset paths and save samples
        check_paths_and_save_samples(val_dataset, "val", 2)
        
        # Create test dataset
        test_dataset = get_segmentation_dataset('tusimple', 
                                               split='test', 
                                               mode='test',
                                               transform=input_transform,
                                               base_size=512,
                                               crop_size=480)
        print(f"✓ Test dataset created successfully with {len(test_dataset)} samples")
        
        # Check test dataset paths and save samples
        check_paths_and_save_samples(test_dataset, "test", 2)
        
        # Test data loading
        if len(train_dataset) > 0:
            img, mask = train_dataset[0]
            print(f"\n✓ Sample data loaded - Image shape: {img.shape}, Mask shape: {mask.shape}")
            print(f"✓ Mask unique values: {torch.unique(mask)}")
            
            # Test with DataLoader
            train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                      batch_size=2, 
                                                      shuffle=True)
            batch_img, batch_mask = next(iter(train_loader))
            print(f"✓ Batch loading works - Batch image shape: {batch_img.shape}, Batch mask shape: {batch_mask.shape}")
        
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test model creation
    try:
        model = get_fast_scnn('tusimple')
        print(f"✓ Model created successfully for TUSimple dataset")
        
        # Check number of output classes
        if hasattr(model, 'module'):
            # For DataParallel models
            classifier = model.module.classifier
        else:
            classifier = model.classifier
        
        # The final conv layer is inside a Sequential in the conv attribute
        final_conv = classifier.conv[1]  # Skip dropout (index 0), get Conv2d (index 1)
        num_classes = final_conv.out_channels
        print(f"✓ Model expects {num_classes} classes")
        
        # Test forward pass
        if len(train_dataset) > 0:
            model.eval()  # Set to evaluation mode to avoid BatchNorm issues
            with torch.no_grad():
                img, mask = train_dataset[0]
                img = img.unsqueeze(0)  # Add batch dimension
                output = model(img)
                print(f"✓ Forward pass successful - Output shape: {output[0].shape}")
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return False
    
    print("\n✓ All tests passed! TUSimple dataset is ready for training.")
    return True

def check_data_files():
    """Check if required data files exist"""
    print("Checking data files...")
    
    base_path = "./manideep1108/tusimple/versions/5/TUSimple"
    required_files = [
        "train_set/seg_label/list/train_val_gt.txt",
        "train_set/seg_label/list/test_gt.txt"
    ]
    
    for file_path in required_files:
        full_path = os.path.join(base_path, file_path)
        if os.path.exists(full_path):
            print(f"✓ Found: {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            return False
    
    return True

if __name__ == '__main__':
    print("=" * 60)
    print("TUSimple Dataset Integration Test")
    print("=" * 60)
    
    # Check if data files exist
    if not check_data_files():
        print("Some required data files are missing. Please check your data directory.")
        exit(1)
    
    # Test dataset
    success = test_tusimple_dataset()
    
    if success:
        print("\n" + "=" * 60)
        print("Ready to train! You can now use:")
        print("python train.py --dataset tusimple --base-size 512 --crop-size 480 --batch-size 4")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("There were some issues. Please check the error messages above.")
        print("=" * 60)
