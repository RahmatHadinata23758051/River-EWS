"""
Comprehensive video testing on ALL videos
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict
import sys

# Import model
import importlib.util
spec = importlib.util.spec_from_file_location("unet_model", Path(__file__).parent / "04_unet_model.py")
unet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unet_module)
create_model = unet_module.create_model


class FloodDetectorOptimal:
    """Flood detection inference dengan trained model"""
    
    def __init__(self, model_path='checkpoints/best_model.pth', device=None, threshold=0.5):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.threshold = threshold
        self.image_size = 256
        
        # Load model
        self.model = create_model(device=self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"✓ Model loaded from: {model_path}")
    
    def process_frame(self, frame):
        """Process single frame"""
        if frame is None:
            return None
        
        original_h, original_w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        frame_resized = cv2.resize(frame_rgb, (self.image_size, self.image_size))
        frame_tensor = torch.from_numpy(frame_resized.astype(np.float32) / 255.0)
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(frame_tensor)
            output_np = output.cpu().numpy()[0, 0]
        
        # Resize back
        mask = cv2.resize(output_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        water_pct = np.sum(mask > self.threshold) / mask.size * 100
        
        # Status
        if water_pct < 5:
            status = "Aman"
        elif water_pct < 15:
            status = "Siaga"
        elif water_pct < 30:
            status = "Waspada"
        else:
            status = "Bahaya"
        
        return {
            'water_percentage': water_pct,
            'status': status,
            'mask': mask
        }
    
    def process_video(self, video_path, save_overlay=False):
        """Process video (returns stats only)"""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return None
        
        # Get properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output if requested
        if save_overlay:
            output_path = Path('video') / f"{video_path.stem}_unet_final.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        else:
            out = None
        
        # Process
        frame_results = []
        status_counts = defaultdict(int)
        water_percentages = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self.process_frame(frame)
            status_counts[result['status']] += 1
            water_percentages.append(result['water_percentage'])
            
            if out:
                # Create overlay
                overlay = frame.copy()
                binary_mask = (result['mask'] > self.threshold).astype(np.uint8) * 255
                water_pixels = binary_mask == 255
                overlay[water_pixels] = [255, 255, 0]  # Cyan
                overlay_blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                
                # Add text
                text = f"Water: {result['water_percentage']:.1f}% | {result['status']}"
                cv2.putText(overlay_blended, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                out.write(overlay_blended)
            
            frame_idx += 1
        
        cap.release()
        if out:
            out.release()
        
        return {
            'video': video_path.name,
            'frames': frame_idx,
            'avg_water': np.mean(water_percentages),
            'max_water': np.max(water_percentages),
            'min_water': np.min(water_percentages),
            'status_dist': dict(status_counts),
            'output': str(Path('video') / f"{video_path.stem}_unet_final.mp4") if save_overlay else None
        }


def main():
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE VIDEO TESTING - ALL VIDEOS")
    print(f"{'='*80}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    detector = FloodDetectorOptimal(device=device)
    
    # Test on all videos
    video_dir = Path('video')
    video_files = sorted([f for f in video_dir.glob('*.mp4') if f.name.endswith('.mp4')])
    
    # Filter out already-processed videos
    video_files = [f for f in video_files if '_detected' not in f.name and '_unet' not in f.name]
    
    print(f"Found {len(video_files)} original videos. Testing all...\n")
    
    all_results = []
    
    for video_file in tqdm(video_files, desc="Testing videos", unit="video"):
        result = detector.process_video(video_file, save_overlay=False)
        if result:
            all_results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"TESTING SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Video Name':30s} {'Frames':>8s} {'Avg Water':>10s} {'Max Water':>10s} {'Dominant Status':<20s}")
    print(f"{'-'*80}")
    
    total_avg = []
    global_status = defaultdict(int)
    
    for result in all_results:
        video_name = result['video'][:30]
        avg_water = result['avg_water']
        max_water = result['max_water']
        
        # Dominant status
        dist = result['status_dist']
        dominant_status = max(dist, key=dist.get)
        dominant_pct = dist[dominant_status] / result['frames'] * 100
        
        total_avg.append(avg_water)
        
        for status, count in dist.items():
            global_status[status] += count
        
        print(f"{video_name:30s} {result['frames']:>8d} {avg_water:>9.1f}% {max_water:>9.1f}% {dominant_status} ({dominant_pct:.0f}%)")
    
    print(f"{'-'*80}\n")
    
    # Global statistics
    print(f"GLOBAL FLOOD STATUS DISTRIBUTION (All Videos Combined):")
    total_frames = sum(global_status.values())
    for status in ['Aman', 'Siaga', 'Waspada', 'Bahaya']:
        count = global_status[status]
        pct = count / total_frames * 100 if total_frames > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {status:8s} : {count:6d} frames ({pct:5.1f}%) {bar}")
    
    print(f"\nOverall Average Water: {np.mean(total_avg):.2f}%")
    print(f"Overall Max Water: {np.max(total_avg):.2f}%")
    print(f"Overall Min Water: {np.min(total_avg):.2f}%")
    
    # Save results
    output_file = Path('test_results_all_videos_unet.json')
    with open(output_file, 'w') as f:
        json.dump({
            'model': 'U-Net (Epoch 17, Val IoU: 0.9408)',
            'device': device,
            'total_videos': len(all_results),
            'total_frames': total_frames,
            'global_stats': {
                'average_water_percentage': round(np.mean(total_avg), 2),
                'max_water_percentage': round(np.max(total_avg), 2),
                'min_water_percentage': round(np.min(total_avg), 2),
                'status_distribution': {k: v for k, v in global_status.items()}
            },
            'video_results': all_results
        }, f, indent=2)
    
    print(f"\n✓ Results saved: {output_file}")


if __name__ == "__main__":
    main()
