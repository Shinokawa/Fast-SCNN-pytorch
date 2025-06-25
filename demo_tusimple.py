"""
TUSimple Lane Segmentation Demo Script
Demonstrates lane detection on TUSimple dataset using trained Fast-SCNN model.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse

from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn


def get_args():
    parser = argparse.ArgumentParser(description='TUSimple Lane Detection Demo')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--image-path', type=str, 
                       help='Path to input image (if not provided, will use dataset samples)')
    parser.add_argument('--output-dir', type=str, default='./demo_output',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for inference')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of dataset samples to process (if --image-path not provided)')
    
    return parser.parse_args()


def load_model(model_path, device):
    """Load trained Fast-SCNN model."""
    print(f"Loading model from {model_path}...")
    model = get_fast_scnn('tusimple', aux=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model


def preprocess_image(image_path, keep_original_size=True):
    """Preprocess input image with option to keep original size."""
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    if keep_original_size:
        # Keep original size - Fast-SCNN supports arbitrary input sizes
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        print(f"Using original image size: {img.size}")
    else:
        # Resize to fixed size (legacy mode)
        transform = transforms.Compose([
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        print(f"Resizing from {img.size} to (480, 480)")
    
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0), img


def postprocess_prediction(pred, original_size, keep_original_size=True):
    """Postprocess model prediction."""
    # Convert to numpy and get class predictions
    pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
    
    if keep_original_size:
        # Model output should already be at original size
        pred_binary = (pred * 255).astype(np.uint8)
        print(f"Prediction shape: {pred.shape}")
    else:
        # Resize to original image size (legacy mode)
        pred_pil = Image.fromarray((pred * 255).astype(np.uint8))
        pred_resized = pred_pil.resize(original_size, Image.NEAREST)
        pred_binary = np.array(pred_resized)
    
    return pred_binary


def create_overlay(original_img, pred_mask, alpha=0.5):
    """Create overlay of prediction on original image."""
    # Convert prediction to colored mask (lane lines in green)
    colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    colored_mask[pred_mask > 127] = [0, 255, 0]  # Green for lane lines
    
    # Convert original image to numpy if needed
    if isinstance(original_img, Image.Image):
        original_np = np.array(original_img)
    else:
        original_np = original_img
    
    # Create overlay
    overlay = original_np.copy()
    mask_area = pred_mask > 127
    overlay[mask_area] = (1 - alpha) * overlay[mask_area] + alpha * colored_mask[mask_area]
    
    return overlay.astype(np.uint8)


def demo_single_image(model, image_path, output_dir, device, keep_original_size=True):
    """Run demo on a single image."""
    print(f"Processing image: {image_path}")
    
    # Preprocess
    img_tensor, original_img = preprocess_image(image_path, keep_original_size)
    img_tensor = img_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        pred = outputs[0]
    
    # Postprocess
    pred_mask = postprocess_prediction(pred, original_img.size, keep_original_size)
    
    # Create overlay
    overlay = create_overlay(original_img, pred_mask)
    
    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save original
    original_img.save(os.path.join(output_dir, f"{base_name}_original.jpg"))
    
    # Save prediction mask
    Image.fromarray(pred_mask).save(os.path.join(output_dir, f"{base_name}_mask.png"))
    
    # Save overlay
    Image.fromarray(overlay).save(os.path.join(output_dir, f"{base_name}_overlay.jpg"))
    
    # Calculate statistics
    lane_pixels = np.sum(pred_mask > 127)
    total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
    lane_ratio = lane_pixels / total_pixels
    
    print(f"Lane pixel ratio: {lane_ratio:.3f}")
    print(f"Results saved to {output_dir}")
    
    return {
        'lane_ratio': lane_ratio,
        'lane_pixels': lane_pixels,
        'total_pixels': total_pixels
    }


def demo_dataset_samples(model, output_dir, device, num_samples=5):
    """Run demo on dataset samples."""
    print(f"Processing {num_samples} samples from TUSimple test dataset...")
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    
    test_dataset = get_segmentation_dataset('tusimple', split='test', mode='test',
                                          transform=transform, base_size=512, crop_size=480)
    
    if len(test_dataset) == 0:
        print("No test samples found!")
        return
    
    # Process samples
    for i in range(min(num_samples, len(test_dataset))):
        # Get sample
        img_tensor, img_name = test_dataset[i]
        
        # Load original image for overlay
        # Find corresponding original image path
        original_path = None
        for root, dirs, files in os.walk('./manideep1108/tusimple/versions/5/TUSimple/test_set/clips'):
            if img_name in files:
                original_path = os.path.join(root, img_name)
                break
        
        if original_path and os.path.exists(original_path):
            original_img = Image.open(original_path).convert('RGB')
        else:
            print(f"Warning: Original image not found for {img_name}")
            continue
        
        # Inference
        img_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            pred = outputs[0]
        
        # Postprocess
        pred_mask = postprocess_prediction(pred, original_img.size)
        
        # Create overlay
        overlay = create_overlay(original_img, pred_mask)
        
        # Save results
        base_name = f"sample_{i:03d}"
        
        # Save original
        original_img.save(os.path.join(output_dir, f"{base_name}_original.jpg"))
        
        # Save prediction mask
        Image.fromarray(pred_mask).save(os.path.join(output_dir, f"{base_name}_mask.png"))
        
        # Save overlay
        Image.fromarray(overlay).save(os.path.join(output_dir, f"{base_name}_overlay.jpg"))
        
        print(f"Processed sample {i+1}/{num_samples}: {img_name}")
    
    print(f"Results saved to {output_dir}")


def main():
    args = get_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, args.device)
    
    if args.image_path:
        # Process single image
        if not os.path.exists(args.image_path):
            print(f"Error: Image file {args.image_path} not found!")
            return
        demo_single_image(model, args.image_path, args.output_dir, args.device)
    else:
        # Process dataset samples
        demo_dataset_samples(model, args.output_dir, args.device, args.num_samples)
    
    print("Demo completed!")


if __name__ == '__main__':
    main()
