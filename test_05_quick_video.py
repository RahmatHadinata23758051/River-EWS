"""
Quick Video Test - Water Detection (No Model Training Required)
Test water detection directly from video using color-based heuristic
"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

class QuickWaterDetector:
    """Water detection menggunakan color-based heuristic"""
    
    def __init__(self):
        # Cyan water color (BGR format)
        self.water_color_bgr = np.array([255, 221, 51], dtype=np.uint8)
        self.color_tolerance = 15  # More lenient for testing
        
    def get_flood_status(self, water_percentage):
        """Classify flood status"""
        if water_percentage < 5:
            return "Aman"
        elif water_percentage < 15:
            return "Siaga"
        elif water_percentage < 30:
            return "Waspada"
        else:
            return "Bahaya"
    
    def process_frame(self, frame):
        """Detect water in single frame"""
        # Detect cyan water pixels
        diff = np.abs(frame.astype(int) - self.water_color_bgr.astype(int))
        water_mask = np.all(diff <= self.color_tolerance, axis=2)
        
        # Calculate percentage
        water_percentage = np.sum(water_mask) / water_mask.size * 100
        
        return water_mask, water_percentage
    
    def process_video(self, video_path, output_path=None, sample_rate=5):
        """
        Process video and detect water
        sample_rate: process every Nth frame for speed
        """
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n{'='*70}")
        print(f"VIDEO: {Path(video_path).name}")
        print(f"{'='*70}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps:.2f}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {total_frames/fps:.2f} seconds")
        print(f"Processing every {sample_rate} frames...")
        
        # Video writer if output is specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_results = []
        frame_idx = 0
        processed_frames = 0
        
        water_pcts = []
        status_counts = defaultdict(int)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Process every Nth frame
            if frame_idx % sample_rate != 0:
                if out:
                    out.write(frame)
                continue
            
            processed_frames += 1
            
            # Detect water
            water_mask, water_pct = self.process_frame(frame)
            status = self.get_flood_status(water_pct)
            
            # Record
            frame_results.append({
                'frame_number': frame_idx,
                'water_percentage': round(water_pct, 2),
                'status': status
            })
            
            water_pcts.append(water_pct)
            status_counts[status] += 1
            
            # Create output frame with overlay
            if out:
                # Red overlay for water
                overlay = frame.copy()
                overlay[water_mask] = [0, 100, 255]
                
                # Blend
                output_frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                
                # Add text
                text = f"Frame {frame_idx:05d} | Water: {water_pct:.1f}% | {status}"
                cv2.putText(output_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 0), 2)
                
                # Status color
                color_status = {
                    'Aman': (0, 255, 0),      # Green
                    'Siaga': (0, 255, 255),   # Yellow
                    'Waspada': (0, 165, 255), # Orange
                    'Bahaya': (0, 0, 255)     # Red
                }
                status_color = color_status.get(status, (255, 255, 255))
                cv2.putText(output_frame, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                           1.0, status_color, 3)
                
                out.write(output_frame)
            
            # Progress
            if processed_frames % 30 == 0:
                print(f"  ✓ Processed {processed_frames} frames / {frame_idx}...")
        
        cap.release()
        if out:
            out.release()
        
        # Statistics
        print(f"\n{'─'*70}")
        print(f"RESULTS:")
        print(f"{'─'*70}")
        print(f"Total frames processed: {processed_frames}")
        print(f"Water detection frames: {sum(1 for r in frame_results if r['water_percentage'] > 5)}")
        print(f"\nAverage water: {np.mean(water_pcts):.2f}%")
        print(f"Max water: {np.max(water_pcts):.2f}%")
        print(f"Min water: {np.min(water_pcts):.2f}%")
        
        print(f"\nFlood Status Distribution:")
        for status in ['Aman', 'Siaga', 'Waspada', 'Bahaya']:
            count = status_counts[status]
            pct = count / len(frame_results) * 100 if frame_results else 0
            print(f"  {status:8s}: {count:3d} frames ({pct:5.1f}%)")
        
        if output_path:
            print(f"\n✓ Output video saved: {output_path}")
        
        return frame_results


def main():
    """Test video processing"""
    
    video_dir = Path('video')
    if not video_dir.exists():
        print("❌ video/ folder not found!")
        return
    
    # List videos
    videos = sorted(video_dir.glob('*.mp4'))
    
    if not videos:
        print("❌ No MP4 videos found!")
        return
    
    print(f"\n{'='*70}")
    print(f"WATER DETECTION TEST - VIDEO ANALYSIS")
    print(f"{'='*70}")
    print(f"Found {len(videos)} videos")
    print(f"\nWhich video to test? (Enter number or 'all')")
    print(f"{'─'*70}")
    
    for i, v in enumerate(videos[:10], 1):
        size_mb = v.stat().st_size / (1024**2)
        print(f"  {i}. {v.name:30s} ({size_mb:6.2f} MB)")
    
    if len(videos) > 10:
        print(f"  ... and {len(videos) - 10} more")
    
    # Test small video first
    test_video = videos[0]  # Start with first (smallest) video
    
    print(f"\n→ Testing with: {test_video.name}")
    
    # Initialize detector
    detector = QuickWaterDetector()
    
    # Output video with detection
    output_video = test_video.parent / f"{test_video.stem}_detected.mp4"
    
    # Process
    try:
        results = detector.process_video(
            test_video,
            output_path=output_video,
            sample_rate=1  # Process all frames
        )
        
        # Save results
        results_file = test_video.parent / f"{test_video.stem}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved: {results_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
