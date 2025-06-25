import os
import argparse
import time
import numpy as np
import torch
import torch.utils.data as data
from torch.cuda.amp import autocast
from torchvision import transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from data_loader import get_segmentation_dataset
from models.fast_scnn import get_fast_scnn
from utils.metric import SegmentationMetric


def parse_args():
    """Testing and Visualization Options"""
    parser = argparse.ArgumentParser(description='Fast-SCNN Testing and Visualization')
    parser.add_argument('--model', type=str, default='fast_scnn')
    parser.add_argument('--dataset', type=str, default='tusimple')
    parser.add_argument('--base-size', type=int, default=520)
    parser.add_argument('--crop-size', type=int, default=480)
    parser.add_argument('--model-path', type=str, default='./weights/fast_scnn_tusimple_best_model.pth',
                        help='path to trained model')
    parser.add_argument('--save-dir', type=str, default='./test_results',
                        help='directory to save test results')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='number of samples to visualize (default: 20)')
    parser.add_argument('--use-fp16', action='store_true', default=True,
                        help='use mixed precision for inference')
    
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    return args


class Tester:
    def __init__(self, args):
        self.args = args
        
        # Create save directory
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Image transform (same as training)
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        
        # Load test dataset
        data_kwargs = {
            'transform': self.input_transform, 
            'base_size': args.base_size, 
            'crop_size': args.crop_size
        }
        self.test_dataset = get_segmentation_dataset(
            args.dataset, split='test', mode='testval', **data_kwargs
        )
        self.test_loader = data.DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Load model
        self.model = get_fast_scnn(dataset=args.dataset, aux=False)
        if os.path.isfile(args.model_path):
            print(f'Loading model from {args.model_path}...')
            state_dict = torch.load(args.model_path, map_location='cpu')
            # Handle DataParallel models
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
        self.model.to(args.device)
        self.model.eval()
        
        # Evaluation metrics
        self.metric = SegmentationMetric(self.test_dataset.num_class)
        
        print(f"\n=== Test Configuration ===")
        print(f"Dataset: {args.dataset}")
        print(f"Model: {args.model}")
        print(f"Model Path: {args.model_path}")
        print(f"Test Dataset Size: {len(self.test_dataset)}")
        print(f"Save Directory: {args.save_dir}")
        print(f"Samples to Visualize: {min(args.num_samples, len(self.test_dataset))}")
        print("=" * 30)
    
    def denormalize_image(self, tensor):
        """Convert normalized tensor back to original image"""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        if tensor.is_cuda:
            mean = mean.cuda()
            std = std.cuda()
            
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return tensor
    
    def create_overlay(self, image, mask, alpha=0.6):
        """Create overlay of lane mask on original image"""
        # Convert tensors to numpy
        if torch.is_tensor(image):
            image = self.denormalize_image(image.cpu())
            image = image.permute(1, 2, 0).numpy()
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        
        # Ensure image is in range [0, 255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Create colored mask (red for lanes)
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 0] = mask * 255  # Red channel for lanes
        
        # Create overlay
        overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
        return image, colored_mask, overlay
    
    def test_and_visualize(self):
        """Run inference on test set and create visualizations"""
        print(f"\n🔍 Starting inference on test set...")
        
        self.metric.reset()
        total_time = 0.0
        visualized_count = 0
        
        with torch.no_grad():
            for i, (image, target) in enumerate(self.test_loader):
                if i >= self.args.num_samples:
                    print(f"Reached maximum samples limit ({self.args.num_samples})")
                    break
                
                start_time = time.time()
                
                image_input = image.to(self.args.device, non_blocking=True)
                target = target.to(self.args.device, non_blocking=True)
                
                # Run inference
                if self.args.use_fp16:
                    with autocast():
                        outputs = self.model(image_input)
                else:
                    outputs = self.model(image_input)
                
                # Get prediction
                if isinstance(outputs, tuple):
                    pred_logits = outputs[0]  # Main output
                else:
                    pred_logits = outputs
                
                pred = torch.argmax(pred_logits, 1)
                inference_time = time.time() - start_time
                total_time += inference_time
                
                # Update metrics
                pred_np = pred.cpu().data.numpy()
                target_np = target.cpu().numpy()
                self.metric.update(pred_np, target_np)
                
                # Create visualization
                original_img = image[0]  # Remove batch dimension
                pred_mask = pred[0].cpu().numpy()  # Remove batch dimension
                target_mask = target[0].cpu().numpy()
                
                # Create overlays
                orig_img, _, pred_overlay = self.create_overlay(original_img, pred_mask)
                _, target_colored, target_overlay = self.create_overlay(original_img, target_mask)
                
                # Create comparison plot
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                fig.suptitle(f'Sample {i+1} - Lane Segmentation Results', fontsize=16)
                
                # Row 1: Original, Ground Truth Mask, Prediction Mask
                axes[0, 0].imshow(orig_img)
                axes[0, 0].set_title('Original Image')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(target_mask, cmap='gray')
                axes[0, 1].set_title('Ground Truth Mask')
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(pred_mask, cmap='gray')
                axes[0, 2].set_title('Predicted Mask')
                axes[0, 2].axis('off')
                
                # Row 2: Overlays and Comparison
                axes[1, 0].imshow(target_overlay)
                axes[1, 0].set_title('Ground Truth Overlay')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(pred_overlay)
                axes[1, 1].set_title('Prediction Overlay')
                axes[1, 1].axis('off')
                
                # Difference visualization
                diff_mask = np.abs(pred_mask.astype(float) - target_mask.astype(float))
                axes[1, 2].imshow(diff_mask, cmap='hot')
                axes[1, 2].set_title('Prediction Error (Red = Wrong)')
                axes[1, 2].axis('off')
                
                # Add metrics text
                pixAcc, mIoU = self.metric.get()
                fig.text(0.02, 0.02, 
                         f'Inference Time: {inference_time*1000:.1f}ms | '
                         f'Current PixAcc: {pixAcc*100:.2f}% | '
                         f'Current mIoU: {mIoU*100:.2f}%',
                         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                
                # Save the plot
                save_path = os.path.join(self.args.save_dir, f'sample_{i+1:03d}_comparison.png')
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                visualized_count += 1
                
                if (i + 1) % 5 == 0:
                    print(f'Processed {i+1} samples | Avg Time: {total_time/(i+1)*1000:.1f}ms | '
                          f'PixAcc: {pixAcc*100:.2f}% | mIoU: {mIoU*100:.2f}%')
        
        # Final metrics
        final_pixAcc, final_mIoU = self.metric.get()
        avg_inference_time = total_time / visualized_count
        
        print(f"\n📊 Test Results Summary:")
        print(f"   Samples Processed: {visualized_count}")
        print(f"   Average Inference Time: {avg_inference_time*1000:.2f}ms")
        print(f"   Final Pixel Accuracy: {final_pixAcc*100:.3f}%")
        print(f"   Final Mean IoU: {final_mIoU*100:.3f}%")
        print(f"   Visualizations saved to: {self.args.save_dir}")
        
        # Create summary plot
        self.create_summary_plot(final_pixAcc, final_mIoU, avg_inference_time, visualized_count)
        
        return {
            'pixAcc': final_pixAcc,
            'mIoU': final_mIoU,
            'avg_time': avg_inference_time,
            'samples': visualized_count
        }
    
    def create_summary_plot(self, pixacc, miou, avg_time, num_samples):
        """Create a summary visualization of test results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Fast-SCNN Test Results Summary', fontsize=16, fontweight='bold')
        
        # Metrics comparison
        metrics = ['Pixel Accuracy', 'Mean IoU']
        values = [pixacc * 100, miou * 100]
        colors = ['skyblue', 'lightcoral']
        
        bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Segmentation Metrics')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # Performance gauge
        ax2.pie([miou, 1-miou], labels=['Achieved', 'Remaining'], 
               colors=['lightgreen', 'lightgray'], startangle=90,
               autopct=lambda pct: f'{miou*100:.1f}%' if pct > 50 else '')
        ax2.set_title(f'mIoU Achievement\n({miou*100:.2f}%)')
        
        # Inference speed
        ax3.bar(['Inference Speed'], [avg_time * 1000], color='orange', alpha=0.8)
        ax3.set_ylabel('Time (ms)')
        ax3.set_title('Average Inference Speed')
        ax3.text(0, avg_time * 1000 + 2, f'{avg_time*1000:.2f}ms', 
                ha='center', va='bottom', fontweight='bold')
        
        # Statistics
        stats_text = f"""
Test Dataset Statistics:
• Samples Processed: {num_samples}
• Pixel Accuracy: {pixacc*100:.3f}%
• Mean IoU: {miou*100:.3f}%
• Avg Inference: {avg_time*1000:.2f}ms
• FPS: {1/avg_time:.1f}

Lane Segmentation Quality:
• {"Excellent" if miou > 0.7 else "Good" if miou > 0.5 else "Fair"} mIoU performance
• {"Excellent" if pixacc > 0.95 else "Good" if pixacc > 0.9 else "Fair"} pixel accuracy
• Real-time capable: {"Yes" if avg_time < 0.1 else "No"}
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Performance Analysis')
        
        plt.tight_layout()
        summary_path = os.path.join(self.args.save_dir, 'test_summary.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Summary plot saved to: {summary_path}")


def main():
    args = parse_args()
    tester = Tester(args)
    results = tester.test_and_visualize()
    
    print(f"\n🎉 Testing and visualization completed!")
    print(f"Results saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
