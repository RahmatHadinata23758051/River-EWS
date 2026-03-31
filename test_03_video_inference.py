"""
Test trained U-Net model on video files
Uses the optimal model (best_model.pth) for water detection
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict
import sys

# Import model dengan cara yang benar
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
        
        # Handle both dict and direct state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Device: {self.device}")
    
    def process_frame(self, frame):
        """Process single frame and return water detection"""
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
            output_np = output.cpu().numpy()[0, 0]  # (256, 256)
        
        # Resize back to original
        mask = cv2.resize(output_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        binary_mask = (mask > self.threshold).astype(np.uint8) * 255
        
        # Calculate water percentage
        water_pct = np.sum(mask > self.threshold) / mask.size * 100
        
        # Determine status
        if water_pct < 5:
            status = "Aman"
        elif water_pct < 15:
            status = "Siaga"
        elif water_pct < 30:
            status = "Waspada"
        else:
            status = "Bahaya"
        
        return {
            'mask': mask,
            'binary_mask': binary_mask,
            'water_percentage': water_pct,
            'status': status
        }
    
    def process_video(self, video_path, output_path=None, max_frames=None, save_overlay=True):
        """Process entire video"""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"✗ Cannot open video: {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video if requested
        if output_path and save_overlay:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        else:
            out = None
        
        # Process frames
        frame_results = []
        status_counts = defaultdict(int)
        water_percentages = []
        
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if max_frames specified
            if max_frames and frame_idx >= max_frames:
                break
            
            # Process frame
            result = self.process_frame(frame)
            
            frame_results.append({
                'frame': frame_idx,
                'water_percentage': round(result['water_percentage'], 2),
                'status': result['status']
            })
            
            status_counts[result['status']] += 1
            water_percentages.append(result['water_percentage'])
            
            # Create and save overlay
            if out:
                overlay = frame.copy()
                mask_color = result['binary_mask']
                
                # Apply water mask overlay (cyan color for water)
                water_pixels = mask_color == 255
                overlay[water_pixels] = [255, 255, 0]  # Cyan in BGR
                
                # Blend
                overlay_blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                
                # Add text
                text = f"Water: {result['water_percentage']:.1f}% | Status: {result['status']}"
                cv2.putText(overlay_blended, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
                out.write(overlay_blended)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        if out:
            out.release()
        
        # Summary
        print(f"\n{'─'*70}")
        print(f"VIDEO ANALYSIS SUMMARY")
        print(f"{'─'*70}")
        print(f"Video: {video_path.name}")
        print(f"Frames processed: {frame_idx}")
        print(f"Average water: {np.mean(water_percentages):.2f}%")
        print(f"Max water: {np.max(water_percentages):.2f}%")
        print(f"Min water: {np.min(water_percentages):.2f}%")
        
        print(f"\nFlood Status Distribution:")
        for status in ['Aman', 'Siaga', 'Waspada', 'Bahaya']:
            count = status_counts[status]
            pct = count / frame_idx * 100 if frame_idx > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"  {status:8s} : {count:4d} frames ({pct:5.1f}%) {bar}")
        
        if output_path:
            print(f"\n✓ Output video saved: {output_path}")
        
        return {
            'video': str(video_path),
            'frames_processed': frame_idx,
            'average_water_percentage': round(np.mean(water_percentages), 2),
            'status_distribution': dict(status_counts),
            'frame_results': frame_results[:10]  # Save first 10 frames for reference
        }


def main():
    print(f"\n{'='*70}")
    print(f"WATER DETECTION - TRAINED NEURAL NETWORK")
    print(f"{'='*70}\n")
    
    # Create detector
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = FloodDetectorOptimal(device=device)
    
    # Test on videos
    video_dir = Path('video')
    video_files = sorted(video_dir.glob('*.mp4'))
    
    if not video_files:
        print("✗ No MP4 videos found in video/ folder!")
        return
    
    print(f"Found {len(video_files)} videos. Testing on first 3...\n")
    
    all_results = []
    for video_file in video_files[:3]:  # Test on first 3 videos
        output_video = video_dir / f"{video_file.stem}_detected_unet.mp4"
        
        result = detector.process_video(video_file, output_path=output_video, max_frames=None)
        if result:
            all_results.append(result)
    
    # Save results
    output_file = Path('test_results_unet_video.json')
    with open(output_file, 'w') as f:
        json.dump({
            'model': 'U-Net (Epoch 17, Val IoU: 0.9408)',
            'device': device,
            'test_date': '2024-03-27',
            'results': all_results
        }, f, indent=2)
    
    print(f"\n✓ Results saved: {output_file}")


if __name__ == "__main__":
    main()
