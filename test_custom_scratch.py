#!/usr/bin/env python3
"""
Test Custom Scratch Model
Test the model trained from scratch on custom dataset
"""
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from PIL import Image
import torchvision.transforms as transforms

def load_model(model_path, device):
    """Load the trained model"""
    from models.fast_scnn import FastSCNN
    
    model = FastSCNN(num_classes=2, aux=True)  # 训练时使用了aux=True，所以加载时也要设为True
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    else:
        print(f"Model file not found: {model_path}")
        return None

def preprocess_image(image_path, target_size=1024):
    """Preprocess image for inference"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize to target size while maintaining aspect ratio
    image = image.resize((target_size, target_size), Image.BILINEAR)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, original_size

def inference_single_image(model, image_path, device, target_size=1024):
    """Run inference on a single image"""
    image_tensor, original_size = preprocess_image(image_path, target_size)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        if isinstance(outputs, tuple):
            pred = outputs[0]
        else:
            pred = outputs
        
        pred = torch.softmax(pred, dim=1)
        pred_mask = torch.argmax(pred, dim=1).cpu().numpy()[0]
    
    return pred_mask, original_size

def visualize_result(image_path, pred_mask, output_path, original_size=None):
    """Visualize the segmentation result"""
    # Load original image
    original_image = cv2.imread(str(image_path))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Resize prediction to match original image size if needed
    if original_size:
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
        # Also resize original image for consistency
        original_image = cv2.resize(original_image, original_size)
    
    # Create colored mask
    colored_mask = np.zeros_like(original_image)
    colored_mask[pred_mask == 1] = [0, 255, 0]  # Green for drivable area
    
    # Create overlay
    overlay = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Green = Drivable)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")

def test_model(model_path, test_images_dir, output_dir, device):
    """Test model on multiple images"""
    # Load model
    model = load_model(model_path, device)
    if model is None:
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test images
    test_images_dir = Path(test_images_dir)
    image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    
    if not image_files:
        print(f"No test images found in {test_images_dir}")
        return
    
    print(f"Found {len(image_files)} test images")
    
    for i, image_path in enumerate(image_files[:10]):  # Test first 10 images
        print(f"Processing {image_path.name}...")
        
        # Run inference
        pred_mask, original_size = inference_single_image(model, image_path, device)
        
        # Calculate statistics
        drivable_pixels = np.sum(pred_mask == 1)
        total_pixels = pred_mask.size
        drivable_ratio = drivable_pixels / total_pixels * 100
        
        print(f"  Drivable area: {drivable_ratio:.2f}% of image")
        
        # Save visualization
        output_path = Path(output_dir) / f"{image_path.stem}_result.png"
        visualize_result(image_path, pred_mask, output_path, original_size)
    
    print(f"\nAll results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Test custom scratch model')
    parser.add_argument('--model-path', default='./weights/custom_scratch/fast_scnn_custom.pth', 
                       help='Path to trained model')
    parser.add_argument('--test-images', default='data/custom/images', 
                       help='Directory containing test images')
    parser.add_argument('--output-dir', default='./custom_scratch_results', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("=" * 60)
    print("Testing Custom Scratch Model")
    print("=" * 60)
    
    test_model(args.model_path, args.test_images, args.output_dir, device)

if __name__ == "__main__":
    main()
