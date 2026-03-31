"""
Test water detection on IMAGES (more reliable than compressed video)
Uses binary masks we already have
"""

import cv2
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

def test_image_detection():
    """Test water detection on sample images"""
    
    images_dir = Path('images/2022111801')
    masks_dir = Path('binary_masks/2022111801')
    
    if not masks_dir.exists():
        print("Binary masks folder not found!")
        print("Please run 03_create_binary_masks.py first")
        return
    
    # Get sample images
    image_files = sorted(list(images_dir.glob('*.jpg')))[:10]
    
    print(f"\n{'='*70}")
    print(f"WATER DETECTION TEST - IMAGES")
    print(f"{'='*70}")
    print(f"Testing {len(image_files)} images from {images_dir.name}")
    
    results = []
    water_percentages = []
    
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        mask_name = img_file.stem + '_binary.png'
        mask_file = masks_dir / mask_name
        
        if not mask_file.exists():
            print(f"⚠️  Mask not found for {img_file.name}")
            continue
        
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        
        # Calculate water percentage
        water_pixels = np.sum(mask == 255)
        total_pixels = mask.size
        water_pct = water_pixels / total_pixels * 100
        
        # Status
        if water_pct < 5:
            status = "Aman"
        elif water_pct < 15:
            status = "Siaga"
        elif water_pct < 30:
            status = "Waspada"
        else:
            status = "Bahaya"
        
        result = {
            'file': img_file.name,
            'water_percentage': round(water_pct, 2),
            'water_pixels': int(water_pixels),
            'total_pixels': total_pixels,
            'status': status
        }
        
        results.append(result)
        water_percentages.append(water_pct)
        
        print(f"  {img_file.name:25s} : {water_pct:6.2f}% water → {status}")
    
    # Summary
    print(f"\n{'─'*70}")
    print(f"SUMMARY:")
    print(f"{'─'*70}")
    print(f"Images tested: {len(results)}")
    print(f"Average water: {np.mean(water_percentages):.2f}%")
    print(f"Max water: {np.max(water_percentages):.2f}%")
    print(f"Min water: {np.min(water_percentages):.2f}%")
    
    # Status distribution
    status_counts = defaultdict(int)
    for r in results:
        status_counts[r['status']] += 1
    
    print(f"\nFlood Status Distribution:")
    for status in ['Aman', 'Siaga', 'Waspada', 'Bahaya']:
        count = status_counts[status]
        pct = count / len(results) * 100 if results else 0
        print(f"  {status:8s} : {count:2d} images ({pct:5.1f}%)")
    
    # Save results
    output_file = Path('test_results_images.json')
    with open(output_file, 'w') as f:
        json.dump({
            'test_date': '2024-03-27',
            'test_source': 'Binary masks from images',
            'total_images': len(results),
            'average_water_percentage': round(np.mean(water_percentages), 2),
            'results': results
        }, f, indent=2)
    
    print(f"\n✓ Results saved: {output_file}")
    
    return results

if __name__ == "__main__":
    test_image_detection()
