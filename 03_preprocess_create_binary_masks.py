"""
Binary Mask Conversion Script
Mengkonversi multi-class masks menjadi binary water masks
"""

import cv2
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

def create_binary_masks():
    """Convert multi-class masks to binary water vs background"""
    
    base_path = Path(".")
    images_path = base_path / "images"
    annotations_path = base_path / "annotations"
    binary_masks_path = base_path / "binary_masks"
    
    # Create output directory
    binary_masks_path.mkdir(exist_ok=True)
    
    # Water class color (BGR format)
    # RGB (51, 221, 255) = BGR (255, 221, 51)
    WATER_COLOR_BGR = np.array([255, 221, 51], dtype=np.uint8)
    COLOR_TOLERANCE = 5  # Pixel value tolerance
    
    print("="*80)
    print("BINARY MASK CONVERSION (Multi-class → Water vs Background)")
    print("="*80)
    
    total_masks_processed = 0
    total_water_pixels = 0
    total_pixels = 0
    
    subfolders = sorted([d for d in images_path.iterdir() if d.is_dir()])
    
    for subfolder in subfolders:
        output_subfolder = binary_masks_path / subfolder.name
        output_subfolder.mkdir(exist_ok=True)
        
        ann_folder = annotations_path / subfolder.name
        if not ann_folder.exists():
            print(f"⚠ Skipping {subfolder.name} - no annotations")
            continue
        
        mask_files = sorted(ann_folder.glob("*.png"))
        
        print(f"\n{subfolder.name}:")
        subfolder_water_pixels = 0
        subfolder_total_pixels = 0
        
        for mask_file in mask_files:
            # Read multi-class mask
            mask = cv2.imread(str(mask_file))
            if mask is None:
                print(f"  ⚠ Failed to read: {mask_file.name}")
                continue
            
            # Create binary mask (water = 255, background = 0)
            # Check if pixel is close to water color (within tolerance)
            diff = np.abs(mask.astype(int) - WATER_COLOR_BGR.astype(int))
            water_mask = np.all(diff <= COLOR_TOLERANCE, axis=2)
            
            # Convert to 8-bit single channel
            binary_mask = np.uint8(water_mask) * 255
            
            # Count pixels
            water_pixels = np.sum(binary_mask) // 255
            total_pixels_in_mask = binary_mask.size
            
            subfolder_water_pixels += water_pixels
            subfolder_total_pixels += total_pixels_in_mask
            
            # Save binary mask
            output_path = output_subfolder / f"{mask_file.stem}_binary.png"
            cv2.imwrite(str(output_path), binary_mask)
            
            total_masks_processed += 1
            total_water_pixels += water_pixels
            total_pixels += total_pixels_in_mask
        
        # Print subfolder stats
        water_percentage = (subfolder_water_pixels / subfolder_total_pixels * 100) if subfolder_total_pixels > 0 else 0
        print(f"  Processed {len(mask_files)} masks")
        print(f"  Water pixels: {subfolder_water_pixels:,} ({water_percentage:.2f}%)")
    
    # Print overall stats
    print("\n" + "="*80)
    print("CONVERSION COMPLETE")
    print("="*80)
    print(f"Total masks processed: {total_masks_processed}")
    print(f"Total water pixels: {total_water_pixels:,}")
    print(f"Total pixels: {total_pixels:,}")
    overall_water_percentage = (total_water_pixels / total_pixels * 100) if total_pixels > 0 else 0
    print(f"Overall water percentage: {overall_water_percentage:.2f}%")
    print(f"\nOutput directory: {binary_masks_path}")
    print(f"Format: 8-bit grayscale PNG (water=255, background=0)")
    
    # Create sample visualization
    print("\nCreating sample visualization...")
    create_visualization_sample(binary_masks_path, images_path, annotations_path)

def create_visualization_sample(binary_masks_path, images_path, annotations_path):
    """Create visualization comparing original and binary masks"""
    try:
        import matplotlib.pyplot as plt
        
        subfolders = sorted([d for d in images_path.iterdir() if d.is_dir()])
        first_folder = subfolders[0]
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Original Multi-class vs Binary Water Masks - {first_folder.name}', fontsize=14)
        
        image_files = sorted(list(first_folder.glob("*.jpg")))[:3]
        
        for i, img_path in enumerate(image_files):
            # Load image
            image = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load original mask
            orig_mask_path = annotations_path / first_folder.name / (img_path.stem + ".png")
            orig_mask = cv2.imread(str(orig_mask_path))
            orig_mask_rgb = cv2.cvtColor(orig_mask, cv2.COLOR_BGR2RGB)
            
            # Load binary mask
            binary_mask_path = binary_masks_path / first_folder.name / f"{img_path.stem}_binary.png"
            binary_mask = cv2.imread(str(binary_mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Display
            axes[i, 0].imshow(image_rgb)
            axes[i, 0].set_title(f'Image {i+1}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(orig_mask_rgb)
            axes[i, 1].set_title(f'Original Multi-class Mask')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(binary_mask, cmap='gray')
            axes[i, 2].set_title(f'Binary Water Mask')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig("visualization_binary_conversion.png", dpi=100, bbox_inches='tight')
        print("[✓] Visualization saved to: visualization_binary_conversion.png")
        
    except Exception as e:
        print(f"Note: Could not create visualization: {e}")

if __name__ == "__main__":
    create_binary_masks()
