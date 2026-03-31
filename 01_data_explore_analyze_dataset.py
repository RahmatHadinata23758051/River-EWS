"""
Dataset Analysis Script - OPTIMIZED
Menganalisis struktur dan karakteristik Flood Segmentation Dataset
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json

def analyze_dataset():
    """Analisis dataset - optimized version"""
    
    base_path = Path(".")
    images_path = base_path / "images"
    annotations_path = base_path / "annotations"
    
    print("=" * 80)
    print("FLOOD SEGMENTATION DATASET ANALYSIS (OPTIMIZED)")
    print("=" * 80)
    
    # 1. Folder Statistics
    print("\n[1] FOLDER STRUCTURE")
    subfolders = sorted([d for d in images_path.iterdir() if d.is_dir()])
    print(f"Total subfolders: {len(subfolders)}")
    print("\nSubfolder Statistics:")
    
    total_images = 0
    stats = {}
    
    for subfolder in subfolders:
        img_count = len(list(subfolder.glob("*.jpg")))
        ann_subfolder = annotations_path / subfolder.name
        ann_count = len(list(ann_subfolder.glob("*.png"))) if ann_subfolder.exists() else 0
        total_images += img_count
        stats[subfolder.name] = {"images": img_count, "annotations": ann_count}
        print(f"  {subfolder.name}: {img_count} images, {ann_count} annotations")
    
    print(f"\nTotal images: {total_images}")
    
    # 2. Image Format Analysis
    print("\n[2] IMAGE FORMAT ANALYSIS")
    sample_image_path = list(subfolders[0].glob("*.jpg"))[0]
    sample_image = cv2.imread(str(sample_image_path))
    print(f"Sample image path: {sample_image_path.name}")
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample image dtype: {sample_image.dtype}")
    
    # 3. Annotation Mask Analysis
    print("\n[3] ANNOTATION MASK ANALYSIS")
    sample_annotation_path = annotations_path / subfolders[0].name / (sample_image_path.stem + ".png")
    mask = cv2.imread(str(sample_annotation_path))
    
    if mask is not None:
        print(f"Sample mask path: {sample_annotation_path.name}")
        print(f"Sample mask shape: {mask.shape}")
        print(f"Sample mask dtype: {mask.dtype}")
        
        # Fast unique colors check
        print("\n[4] UNIQUE COLORS IN MASKS (Sampled)")
        unique_colors = set()
        
        # Check first 3 folders, first 3 masks each
        for subfolder in subfolders[:3]:
            ann_folder = annotations_path / subfolder.name
            if ann_folder.exists():
                print(f"  Scanning {subfolder.name}...", end=" ")
                for mask_file in list(ann_folder.glob("*.png"))[:3]:
                    m = cv2.imread(str(mask_file))
                    if m is not None:
                        # Get unique rows/pixels as RGB tuples
                        unique_pixels = np.unique(m.reshape(-1, 3), axis=0)
                        for pixel in unique_pixels:
                            rgb = (pixel[2], pixel[1], pixel[0])  # BGR to RGB
                            unique_colors.add(rgb)
                print(f"{len(unique_colors)} unique colors so far")
        
        print(f"\nTotal unique colors found: {len(unique_colors)}")
        print("\nSample colors (first 15, in RGB format):")
        for i, rgb in enumerate(sorted(list(unique_colors))[:15]):
            print(f"  {i+1}. RGB{rgb}")
    
    # 4. Data Distribution
    print("\n[5] DATA DISTRIBUTION")
    print("\nImages per folder:")
    img_counts = [stats[f]["images"] for f in stats if stats[f]["images"] > 0]
    print(f"  Min: {min(img_counts) if img_counts else 0}")
    print(f"  Max: {max(img_counts) if img_counts else 0}")
    print(f"  Mean: {np.mean(img_counts):.2f}" if img_counts else "  Mean: 0")
    print(f"  Median: {np.median(img_counts):.2f}" if img_counts else "  Median: 0")
    
    # 5. Save analysis report
    report = {
        "total_subfolders": len(subfolders),
        "total_images": total_images,
        "subfolder_stats": stats,
        "sample_image_shape": list(sample_image.shape) if sample_image is not None else None,
        "sample_mask_shape": list(mask.shape) if mask is not None else None,
        "unique_colors_count": len(unique_colors) if 'unique_colors' in locals() else 0,
    }
    
    with open("dataset_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n[✓] Analysis report saved to: dataset_analysis_report.json")
    
    # 6. Visualization Info
    print("\n[6] NEXT STEPS")
    print("  1. Visualize sample images and masks")
    print("  2. Identify water class color")
    print("  3. Convert multi-class masks to binary water masks")
    print("  4. Prepare dataset split (train/val/test)")
    print("  5. Build U-Net model")
    print("  6. Train segmentation model")

if __name__ == "__main__":
    analyze_dataset()

