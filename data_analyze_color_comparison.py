"""
Compare colors from images vs videos
"""

import cv2
import numpy as np
from pathlib import Path
import random

def analyze_image_colors(image_path):
    """Analyze colors in single image"""
    img = cv2.imread(str(image_path))
    if img is None:
        return {}
    
    img_2d = img.reshape(-1, 3)
    
    # Sample pixels
    indices = np.random.choice(img_2d.shape[0], min(1000, img_2d.shape[0]), replace=False)
    colors = {}
    
    for pixel in img_2d[indices]:
        rgb = (pixel[2], pixel[1], pixel[0])  # BGR to RGB
        key = str(rgb)
        colors[key] = colors.get(key, 0) + 1
    
    return colors

def main():
    image_dir = Path('images')
    images = list(image_dir.rglob('*.jpg'))[:5]  # First 5 images
    
    print(f"\n{'='*70}")
    print(f"ANALYZING IMAGE COLORS")
    print(f"{'='*70}")
    
    all_color_counts = {}
    
    for img_path in images:
        print(f"\n{img_path.name}:")
        colors = analyze_image_colors(img_path)
        
        # Show top colors
        sorted_colors = sorted(colors.items(), key=lambda x: x[1], reverse=True)
        for color_str, count in sorted_colors[:5]:
            print(f"  {color_str} : {count} pixels")
        
        # Aggregate
        for color_str, count in colors.items():
            all_color_counts[color_str] = all_color_counts.get(color_str, 0) + count
    
    print(f"\n{'─'*70}")
    print("TOP COLORS ACROSS ALL IMAGES:")
    print(f"{'─'*70}")
    
    sorted_all = sorted(all_color_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (color_str, count) in enumerate(sorted_all[:20], 1):
        print(f"{i:2d}. {color_str} : {count:6d} pixels")
    
    # Check for cyan specifically
    print(f"\n{'─'*70}")
    print("WATER COLOR DETECTION:")
    print(f"Looking for RGB (51, 221, 255) [Cyan]...")
    
    cyan_exact = "(51, 221, 255)"
    if cyan_exact in all_color_counts:
        print(f"  ✓ Found: {all_color_counts[cyan_exact]} pixels")
    else:
        print(f"  ✗ Not found exactly")
        
        # Find similar blues
        for color_str in all_color_counts.keys():
            try:
                rgb = eval(color_str)
                # Look for high blue/green values
                if rgb[2] > 200 and rgb[1] > 200 and rgb[0] < 100:  # High B+G, low R
                    pct = all_color_counts[color_str] / sum(all_color_counts.values()) * 100
                    print(f"    Similar: {color_str} ({pct:.2f}%)")
            except:
                pass

if __name__ == "__main__":
    main()
