"""
Comprehensive test on ALL binary masks
Shows flood status distribution across entire dataset
"""

import cv2
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

def test_all_masks():
    """Test water detection on all binary masks"""
    
    binary_masks_dir = Path('binary_masks')
    
    if not binary_masks_dir.exists():
        print("Binary masks folder not found!")
        return
    
    folders = sorted([f for f in binary_masks_dir.iterdir() if f.is_dir()])
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE WATER DETECTION TEST")
    print(f"{'='*70}")
    print(f"Testing binary masks from {len(folders)} folders\n")
    
    all_results = []
    folder_stats = {}
    
    for folder in folders:
        mask_files = sorted(folder.glob('*_binary.png'))
        
        if not mask_files:
            continue
        
        print(f"Processing {folder.name} : {len(mask_files)} masks...", end=' ', flush=True)
        
        water_percentages = []
        status_counts = defaultdict(int)
        
        for mask_file in mask_files:
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
            
            water_percentages.append(water_pct)
            status_counts[status] += 1
            
            all_results.append({
                'folder': folder.name,
                'file': mask_file.stem,
                'water_percentage': round(water_pct, 2),
                'status': status
            })
        
        avg_water = np.mean(water_percentages)
        folder_stats[folder.name] = {
            'mask_count': len(mask_files),
            'average_water': round(avg_water, 2),
            'max_water': round(np.max(water_percentages), 2),
            'min_water': round(np.min(water_percentages), 2),
            'status_distribution': dict(status_counts)
        }
        
        print(f"✓ avg={avg_water:.1f}% → {dict(status_counts)}")
    
    # Overall summary
    print(f"\n{'─'*70}")
    print(f"FOLDER SUMMARY:")
    print(f"{'─'*70}")
    print(f"{'Folder':20s} {'Images':>8s} {'Water %':>10s} {'Status Distribution':<30s}")
    print(f"{'─'*70}")
    
    total_images = 0
    status_totals = defaultdict(int)
    
    for folder_name in sorted(folder_stats.keys()):
        stats = folder_stats[folder_name]
        total_images += stats['mask_count']
        
        dist = stats['status_distribution']
        for status in ['Aman', 'Siaga', 'Waspada', 'Bahaya']:
            status_totals[status] += dist.get(status, 0)
        
        status_str = ', '.join([f"{s}:{c}" for s, c in dist.items()])
        
        print(f"{folder_name:20s} {stats['mask_count']:>8d} {stats['average_water']:>9.1f}% {status_str:<30s}")
    
    # Overall statistics
    print(f"{'─'*70}")
    print(f"{'TOTAL':20s} {total_images:>8d}")
    print(f"\nOverall Flood Status Distribution:")
    for status in ['Aman', 'Siaga', 'Waspada', 'Bahaya']:
        count = status_totals[status]
        pct = count / total_images * 100 if total_images else 0
        bar = "█" * int(pct / 2)
        print(f"  {status:8s} : {count:4d} images ({pct:5.1f}%) {bar}")
    
    # Save detailed results
    output_file = Path('test_results_all_masks.json')
    with open(output_file, 'w') as f:
        json.dump({
            'test_date': '2024-03-27',
            'test_source': 'All binary masks',
            'total_folders': len(folders),
            'total_images': total_images,
            'folder_stats': folder_stats,
            'status_distribution': dict(status_totals)
        }, f, indent=2)
    
    print(f"\n✓ Detailed results saved: {output_file}")
    
    return all_results, folder_stats

if __name__ == "__main__":
    test_all_masks()
