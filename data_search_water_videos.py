"""
Find videos with actual water detection
"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

def quick_check_video(video_path):
    """Quick check if video has cyan water"""
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // 3)  # Check 3 frames
    
    cyan_pixel_count = 0
    total_pixels = 0
    
    frame_idx = 0
    checked = 0
    
    while checked < 3:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        if frame_idx % frame_interval != 0 and checked < 3:
            continue
        
        checked += 1
        
        # Detect cyan (255, 221, 51) ± 20
        cyan_color = np.array([255, 221, 51], dtype=np.uint8)
        diff = np.abs(frame.astype(int) - cyan_color.astype(int))
        cyan_mask = np.all(diff <= 20, axis=2)
        
        cyan_pixel_count += np.sum(cyan_mask)
        total_pixels += cyan_mask.size
    
    cap.release()
    
    cyan_pct = cyan_pixel_count / total_pixels * 100 if total_pixels > 0 else 0
    return cyan_pct

def main():
    video_dir = Path('video')
    videos = sorted(video_dir.glob('*.mp4'))
    
    print(f"\n{'='*70}")
    print(f"SCANNING VIDEOS FOR WATER DETECTION")
    print(f"{'='*70}")
    
    results = []
    for video in videos:
        cyan_pct = quick_check_video(video)
        size_mb = video.stat().st_size / (1024**2)
        results.append((video.name, size_mb, cyan_pct))
        print(f"  {video.name:30s} : {cyan_pct:6.2f}% water | {size_mb:6.2f} MB")
    
    print(f"\n{'─'*70}")
    print("Videos with water detected:")
    results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
    
    for name, size, pct in results_sorted[:10]:
        if pct > 0:
            print(f"  {name:30s} : {pct:6.2f}% cyan | {size:6.2f} MB ✓")
    
    # Recommend best video for testing
    if results_sorted[0][2] > 0:
        best = results_sorted[0]
        print(f"\n💡 Best video for testing: {best[0]}")
    else:
        print(f"\n⚠️  No videos found with cyan water!")

if __name__ == "__main__":
    main()
