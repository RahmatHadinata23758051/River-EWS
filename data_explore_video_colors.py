"""
Analyze video frame colors to find actual water color
"""

import cv2
import numpy as np
from pathlib import Path

def analyze_video_colors(video_path, num_frames=10):
    """Extract and analyze colors from video frames"""
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)
    
    print(f"\n{'='*70}")
    print(f"ANALYZING: {Path(video_path).name}")
    print(f"{'='*70}")
    print(f"Total frames: {total_frames}")
    print(f"Sampling every {frame_interval} frames...")
    
    all_colors = {}
    frame_idx = 0
    sampled = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue
        
        sampled += 1
        
        # Reshape to 2D and get unique colors
        h, w, c = frame.shape
        frame_2d = frame.reshape(-1, 3)
        
        # Sample random pixels to speed up
        indices = np.random.choice(frame_2d.shape[0], min(1000, frame_2d.shape[0]), replace=False)
        
        for pixel in frame_2d[indices]:
            color_key = tuple(pixel.astype(int))
            all_colors[color_key] = all_colors.get(color_key, 0) + 1
    
    cap.release()
    
    # Top colors
    print(f"\nTop 20 most frequent colors (BGR):")
    print(f"{'─'*70}")
    sorted_colors = sorted(all_colors.items(), key=lambda x: x[1], reverse=True)
    
    for i, (bgr, count) in enumerate(sorted_colors[:20], 1):
        rgb = (bgr[2], bgr[1], bgr[0])
        pct = count / sum(c for _, c in sorted_colors) * 100
        print(f"{i:2d}. BGR{str(bgr):20s} → RGB{str(rgb):20s} : {pct:6.2f}% ({count:8d} pixels)")
    
    print(f"\n{'─'*70}")
    print("💡 TIP: Look for cyan/water-like colors in the list above")
    print("     Current cyan detection: BGR(255, 221, 51) = RGB(51, 221, 255)")


if __name__ == "__main__":
    video_path = Path('video/2022111801.mp4')
    analyze_video_colors(video_path, num_frames=5)
