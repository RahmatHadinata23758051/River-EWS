#!/usr/bin/env python3
"""
QUICK TEST - Flood Detection Model
Simple testing script untuk cepat cek model
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys

# Import model
import importlib.util
spec = importlib.util.spec_from_file_location("unet_model", Path(__file__).parent / "04_unet_model.py")
unet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unet_module)
create_model = unet_module.create_model


def test_model():
    """Test model on sample image"""
    
    print("\n" + "="*70)
    print("FLOOD DETECTION MODEL - QUICK TEST")
    print("="*70 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Load model
    print("Loading model...")
    model = create_model(device=device)
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully!\n")
    
    # Find test image
    image_dir = Path('images/2022111801')
    image_files = sorted(list(image_dir.glob('*.jpg')))
    
    if not image_files:
        print("ERROR: No test images found in images/2022111801/")
        return
    
    print(f"Testing on {len(image_files)} images from {image_dir.name}/\n")
    
    # Process images
    results = []
    for idx, img_file in enumerate(image_files[:5]):  # Test first 5
        image = cv2.imread(str(img_file))
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        resized = cv2.resize(rgb, (256, 256))
        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(tensor)
            output_np = output.cpu().numpy()[0, 0]
        
        # Calculate water
        water_pct = np.sum(output_np > 0.5) / output_np.size * 100
        
        # Status
        if water_pct < 5:
            status = "Aman"
        elif water_pct < 15:
            status = "Siaga"
        elif water_pct < 30:
            status = "Waspada"
        else:
            status = "Bahaya"
        
        results.append((img_file.name, water_pct, status))
        print(f"  [{idx+1}] {img_file.name:25s} → {water_pct:6.2f}% water [{status}]")
    
    # Summary
    print("\n" + "-"*70)
    water_pcts = [r[1] for r in results]
    print(f"Average water: {np.mean(water_pcts):.2f}%")
    print(f"Max water: {np.max(water_pcts):.2f}%")
    print(f"Min water: {np.min(water_pcts):.2f}%")
    
    print("\n" + "="*70)
    print("MODEL STATUS: OK")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        test_model()
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
