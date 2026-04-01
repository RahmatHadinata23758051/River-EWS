"""
VISUALIZATION - Model Results
Visualize water detection results dengan berbagai format
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import json

import importlib.util
spec = importlib.util.spec_from_file_location("unet_model", Path(__file__).parent / "04_model_unet_architecture.py")
unet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unet_module)
create_model = unet_module.create_model


class Visualizer:
    """Visualize model predictions"""
    
    def __init__(self, model_path='checkpoints/best_model.pth', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = create_model(device=self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        self.threshold = 0.5
    
    def predict_frame(self, frame):
        """Get prediction mask"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        resized = cv2.resize(rgb, (256, 256))
        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            output_np = output.cpu().numpy()[0, 0]
        
        mask = cv2.resize(output_np, (w, h), interpolation=cv2.INTER_LINEAR)
        binary = (mask > self.threshold).astype(np.uint8) * 255
        
        water_pct = np.sum(mask > self.threshold) / mask.size * 100
        
        return mask, binary, water_pct
    
    def visualize_comparison(self, image_path, mask_path=None):
        """Side-by-side: Image + Prediction + Ground Truth (if available)"""
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Cannot load image: {image_path}")
            return
        
        pred_mask, pred_binary, water_pct = self.predict_frame(image)
        
        # Prepare subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Water Detection: {Path(image_path).name}\nWater: {water_pct:.1f}%', 
                     fontsize=14, fontweight='bold')
        
        # [0,0] Original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # [0,1] Prediction probability map
        im = axes[0, 1].imshow(pred_mask, cmap='coolwarm', vmin=0, vmax=1)
        axes[0, 1].set_title('Model Prediction (Probability)')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1])
        
        # [1,0] Prediction binary mask
        axes[1, 0].imshow(pred_binary, cmap='gray')
        axes[1, 0].set_title(f'Binary Mask (threshold=0.5)')
        axes[1, 0].axis('off')
        
        # [1,1] Overlay on original
        overlay = image.copy()
        water_pixels = pred_binary == 255
        overlay[water_pixels] = [255, 255, 0]  # Cyan in BGR
        overlay_blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        axes[1, 1].imshow(cv2.cvtColor(overlay_blended, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Overlay (Cyan = Water)')
        axes[1, 1].axis('off')
        
        # Ground truth if available
        if mask_path:
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                fig.set_size_inches(16, 10)
                fig.delaxes(axes[1, 1])
                
                # Add GT visualization
                ax_gt = fig.add_subplot(2, 3, 6)
                ax_gt.imshow(gt_mask, cmap='gray')
                ax_gt.set_title('Ground Truth Mask')
                ax_gt.axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_batch(self, folder_path, num_samples=6):
        """Visualize multiple images from folder"""
        
        folder = Path(folder_path)
        image_files = sorted(list(folder.glob('*.jpg')))[:num_samples]
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        fig.suptitle(f'Water Detection Results - {folder.name}', fontsize=14, fontweight='bold')
        
        for idx, img_file in enumerate(image_files):
            image = cv2.imread(str(img_file))
            pred_mask, pred_binary, water_pct = self.predict_frame(image)
            
            # Original
            axes[idx, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[idx, 0].set_title(f'{img_file.name}\n{water_pct:.1f}% water')
            axes[idx, 0].axis('off')
            
            # Prediction
            axes[idx, 1].imshow(pred_mask, cmap='hot', vmin=0, vmax=1)
            axes[idx, 1].set_title('Prediction')
            axes[idx, 1].axis('off')
            
            # Overlay
            overlay = image.copy()
            overlay[pred_binary == 255] = [255, 255, 0]
            overlay_blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
            axes[idx, 2].imshow(cv2.cvtColor(overlay_blended, cv2.COLOR_BGR2RGB))
            axes[idx, 2].set_title('Overlay')
            axes[idx, 2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_statistics(self, folders):
        """Visualize water percentage distribution across folders"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Water Detection Statistics', fontsize=14, fontweight='bold')
        
        all_water_pcts = []
        folder_stats = {}
        
        for folder in folders:
            folder_path = Path(folder)
            if not folder_path.exists():
                continue
            
            water_pcts = []
            image_files = sorted(folder_path.glob('*.jpg'))[:50]  # Sample 50 images
            
            for img_file in tqdm(image_files, desc=f"Processing {folder_path.name}"):
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                
                _, _, water_pct = self.predict_frame(image)
                water_pcts.append(water_pct)
            
            if water_pcts:
                folder_stats[folder_path.name] = {
                    'mean': np.mean(water_pcts),
                    'std': np.std(water_pcts),
                    'min': np.min(water_pcts),
                    'max': np.max(water_pcts)
                }
                all_water_pcts.extend(water_pcts)
        
        # [0,0] Histogram
        axes[0, 0].hist(all_water_pcts, bins=30, color='steelblue', edgecolor='black')
        axes[0, 0].set_xlabel('Water Percentage (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Water Detection')
        axes[0, 0].axvline(np.mean(all_water_pcts), color='red', linestyle='--', label='Mean')
        axes[0, 0].legend()
        
        # [0,1] Box plot by folder
        if folder_stats:
            folders_list = []
            means = []
            for fname, stats in folder_stats.items():
                folders_list.append(fname[:15])
                means.append(stats['mean'])
            
            axes[0, 1].bar(range(len(folders_list)), means, color='steelblue', edgecolor='black')
            axes[0, 1].set_xticks(range(len(folders_list)))
            axes[0, 1].set_xticklabels(folders_list, rotation=45, ha='right')
            axes[0, 1].set_ylabel('Mean Water (%)')
            axes[0, 1].set_title('Average Water by Folder')
        
        # [1,0] Status distribution
        status_counts = {
            'Aman': sum(1 for w in all_water_pcts if w < 5),
            'Siaga': sum(1 for w in all_water_pcts if 5 <= w < 15),
            'Waspada': sum(1 for w in all_water_pcts if 15 <= w < 30),
            'Bahaya': sum(1 for w in all_water_pcts if w >= 30)
        }
        
        colors = ['green', 'yellow', 'orange', 'red']
        axes[1, 0].pie(status_counts.values(), labels=status_counts.keys(), 
                       colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Flood Status Distribution')
        
        # [1,1] Statistics table
        axes[1, 1].axis('off')
        stats_text = "Statistics:\n\n"
        stats_text += f"Total samples: {len(all_water_pcts)}\n"
        stats_text += f"Mean water: {np.mean(all_water_pcts):.2f}%\n"
        stats_text += f"Std dev: {np.std(all_water_pcts):.2f}%\n"
        stats_text += f"Min water: {np.min(all_water_pcts):.2f}%\n"
        stats_text += f"Max water: {np.max(all_water_pcts):.2f}%\n\n"
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        return fig


def main():
    print("\n" + "="*80)
    print("WATER DETECTION VISUALIZATION")
    print("="*80 + "\n")
    
    viz = Visualizer()
    
    # Test 1: Single image comparison
    print("1. Visualizing single image...")
    fig1 = viz.visualize_comparison('images/2022111801/2022111801_000.jpg')
    fig1.savefig('viz_single_image.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: viz_single_image.png\n")
    
    # Test 2: Batch visualization
    print("2. Visualizing batch (6 samples)...")
    fig2 = viz.visualize_batch('images/2022111801')
    fig2.savefig('viz_batch_images.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: viz_batch_images.png\n")
    
    # Test 3: Statistics
    print("3. Computing statistics...")
    folders = [
        'images/2022111801',
        'images/2022111802',
        'images/2022111803'
    ]
    fig3 = viz.visualize_statistics(folders)
    fig3.savefig('viz_statistics.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: viz_statistics.png\n")
    
    print("="*80)
    print("Visualization files saved!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
