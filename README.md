# 🌊 FLOOD DETECTION MODEL

## Quick Start

### 1. Test Model (Fastest)
```bash
python quick_test.py
```
Tests model on 5 sample images and shows water detection results.

### 2. Test on Images
```bash
python inference.py images/2022111801/2022111801_000.jpg
```

### 3. Test on Videos
```bash
python inference.py video/2022111801.mp4
```

---

## Model Information

- **Architecture**: U-Net Semantic Segmentation
- **Validation IoU**: 94.08%
- **Model Size**: 29.7 MB
- **Parameters**: 7.77M
- **Tested on**: 20 videos, 12,981 frames

---

## Flood Status Classification

| Status | Water % | Color |
|--------|---------|-------|
| Aman | <5% | Green |
| Siaga | 5-15% | Yellow |
| Waspada | 15-30% | Orange |
| Bahaya | >30% | Red |

---

## Dataset

- **3,574 images** (1280×720)
- **1,396 binary masks** created from multi-class annotations
- **20 video files** for testing
- Water class (Cyan): 69-72% of flood pixels

---

## File Structure

```
flood_dataset/
├── quick_test.py              ← START HERE (simple test)
├── inference.py               (image/video inference)
├── 04_unet_model.py          (model architecture)
├── 05_train.py               (training script)
├── README.md                 (this file)
│
├── checkpoints/
│   └── best_model.pth        (trained model)
│
├── flood_detection_model/    (production package)
│   ├── model/best_model.pth
│   ├── code/
│   └── setup.py
│
├── images/                   (3,574 images)
├── videos/                   (20 test videos)
├── annotations/              (segmentation masks)
└── binary_masks/             (1,396 water masks)
```

---

## Testing Results

**Global Flood Status Distribution (20 videos, 12,981 frames):**
- **Bahaya** (Danger): 47.8%
- **Waspada** (Warning): 22.1%
- **Siaga** (Alert): 15.8%
- **Aman** (Safe): 14.3%

---

## Next Steps

1. **For model inference**: Use `quick_test.py` or `inference.py`
2. **For integration**: Use model from `flood_detection_model/`
3. **For retraining**: Use `05_train.py`

---

## Requirements

- Python 3.8+
- PyTorch 2.1.2
- OpenCV 4.8+
- NumPy

Install dependencies:
```bash
pip install -r requirements.txt
```

---

**Status**: ✅ Production Ready | Model trained and validated
