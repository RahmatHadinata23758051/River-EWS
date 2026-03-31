"""
Visualization Script
Menampilkan sample images dengan masks untuk identifikasi water class
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import ListedColormap

def visualize_samples():
    """Visualisasi sample images dan annotations"""
    
    base_path = Path(".")
    images_path = base_path / "images"
    annotations_path = base_path / "annotations"
    
    # Get first folder
    subfolders = sorted([d for d in images_path.iterdir() if d.is_dir()])
    if not subfolders:
        print("No subfolders found!")
        return
    
    first_folder = subfolders[0]
    print(f"Visualizing from: {first_folder.name}")
    
    # Get first 6 image-mask pairs
    image_files = sorted(list(first_folder.glob("*.jpg")))[:6]
    
    fig, axes = plt.subplots(2, 6, figsize=(20, 8))
    fig.suptitle(f'Sample Images and Segmentation Masks from {first_folder.name}', fontsize=16)
    
    for i, img_path in enumerate(image_files):
        # Load image
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load corresponding mask
        mask_path = annotations_path / first_folder.name / (img_path.stem + ".png")
        mask = cv2.imread(str(mask_path))
        
        if mask is None:
            print(f"Warning: Mask not found for {img_path.name}")
            continue
        
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # Display image
        axes[0, i].imshow(image_rgb)
        axes[0, i].set_title(f'{img_path.name}\n({image.shape[1]}x{image.shape[0]})')
        axes[0, i].axis('off')
        
        # Display mask
        axes[1, i].imshow(mask_rgb)
        axes[1, i].set_title(f'Mask - {mask_path.name}')
        axes[1, i].axis('off')
        
        # Print mask color info
        unique_colors = np.unique(mask.reshape(-1, 3), axis=0)
        print(f"\n{img_path.name}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Unique colors in mask: {len(unique_colors)}")
        for j, color in enumerate(unique_colors):
            color_rgb = (color[2], color[1], color[0])
            pixel_count = np.sum(np.all(mask == color, axis=2))
            percentage = (pixel_count / mask.size * 3) * 100
            print(f"    Color {j+1}: BGR{tuple(color)} / RGB{color_rgb} - {pixel_count} pixels ({percentage:.2f}%)")
    
    plt.tight_layout()
    plt.savefig("visualization_samples.png", dpi=100, bbox_inches='tight')
    print("\n[✓] Visualization saved to: visualization_samples.png")
    
    # Create color legend
    print("\n" + "="*60)
    print("COLOR LEGEND ANALYSIS")
    print("="*60)
    
    # Analyze all colors across dataset
    all_colors = {}
    print("\nAnalyzing color distribution across dataset...")
    
    for folder in subfolders[:5]:  # Sample first 5 folders
        masks_path = annotations_path / folder.name
        if not masks_path.exists():
            continue
        
        for mask_file in list(masks_path.glob("*.png"))[:2]:  # 2 masks per folder
            mask = cv2.imread(str(mask_file))
            if mask is not None:
                unique = np.unique(mask.reshape(-1, 3), axis=0)
                for color in unique:
                    color_key = tuple(color)
                    if color_key not in all_colors:
                        all_colors[color_key] = 0
                    all_colors[color_key] += 1
    
    print("\nColor Summary (BGR → RGB, sorted by frequency):")
    sorted_colors = sorted(all_colors.items(), key=lambda x: x[1], reverse=True)
    
    color_names = {
        (0, 0, 0): "Black - Background",
        (255, 221, 51): "Cyan - Likely WATER",  # BGR to RGB
        (245, 61, 61): "Blue - Likely Vegetation",
        (162, 179, 92): "Teal - Building/Road",
        (102, 255, 102): "Green - Vegetation",
        (83, 50, 250): "Red/Pink - Human/Object",
        (55, 96, 255): "Orange - Road/Additional",
    }
    
    for (bgr), count in sorted_colors:
        rgb = (bgr[2], bgr[1], bgr[0])  # Convert to RGB
        name = "Unknown"
        # Find closest match in common colors
        for known_bgr, description in color_names.items():
            if all(abs(int(a) - int(b)) < 5 for a, b in zip(bgr, known_bgr)):
                name = description
                break
        print(f"  BGR{bgr} → RGB{rgb}: {count} occurrences - {name}")

if __name__ == "__main__":
    visualize_samples()
    print("\nYou can now review the images to identify the water class color!")
