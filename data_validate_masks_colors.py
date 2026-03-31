"""
Analyze mask annotation colors directly
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_mask_colors():
    """Analyze colors in annotation masks"""
    
    mask_dir = Path('annotations/2022111801')
    if not mask_dir.exists():
        print("Annotations folder not found!")
        return
    
    mask_files = sorted(list(mask_dir.glob('*.png')))[:5]
    
    print(f"\n{'='*70}")
    print(f"ANALYZING ANNOTATION MASK COLORS")
    print(f"{'='*70}")
    
    all_colors = {}
    
    for mask_file in mask_files:
        print(f"\n{mask_file.name}:")
        mask = cv2.imread(str(mask_file))
        
        if mask is None:
            print(f"  Failed to load")
            continue
        
        # Get unique colors
        h, w, c = mask.shape
        mask_2d = mask.reshape(-1, 3)
        unique_pixels = np.unique(mask_2d, axis=0)
        
        print(f"  Shape: {h}x{w}, Unique colors: {len(unique_pixels)}")
        
        for pixel in unique_pixels:
            bgr = tuple(pixel.astype(int))
            rgb = (bgr[2], bgr[1], bgr[0])
            key = str(rgb)
            all_colors[key] = all_colors.get(key, 0) + 1
            
            # Count pixels of this color
            color_count = np.sum(np.all(mask_2d == pixel, axis=1))
            pct = color_count / mask_2d.shape[0] * 100
            print(f"    RGB{rgb} : {pct:6.2f}%")
    
    print(f"\n{'─'*70}")
    print("MASTER COLOR LIST (All masks):")
    sorted_colors = sorted(all_colors.items(), key=lambda x: x[1], reverse=True)
    
    for color_str, count in sorted_colors[:15]:
        print(f"  {color_str} : {count} occurrences")
    
    # Check for cyan
    print(f"\n{'─'*70}")
    print("WATER COLOR (Cyan):")
    
    cyan_str = "(51, 221, 255)"
    if cyan_str in all_colors:
        print(f"  ✓ Found Cyan {cyan_str}!")
    else:
        print(f"  ✗ Cyan not found")
        print("  Looking for similar blue colors...")
        
        for color_str in sorted_colors[:10]:
            print(f"    {color_str}")

if __name__ == "__main__":
    analyze_mask_colors()
