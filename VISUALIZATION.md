# 📊 VISUALIZATION RESULTS - SUDAH DIBUAT

## Files PNG Generated ✅

```
✓ viz_single_image.png      → 4-panel comparison (image + predictions + overlay)
✓ viz_batch_images.png      → 6 sample images dengan predictions
✓ viz_statistics.png         → Distribution histogram & status pie chart
```

## Cara Menggunakan

### **OPSI 1: View PNG yang Sudah Ada** (Paling Cepat)
```bash
# Lihat file PNG di folder ini
# Atau double-click direktnya
```

### **OPSI 2: Interactive Dashboard**
```bash
python viz_dashboard.py

Menu pilihan:
[1] Visualize single image
[2] Visualize batch images (6 samples)
[3] Visualize statistics
[4] Visualize video results (perlu test video dulu)
[5] Run all visualizations
[6] Open folder with explorer
[0] Exit
```

### **OPSI 3: Script Langsung**

**Test single image:**
```bash
python visualize_results.py
# Auto-generates viz_single_image.png, viz_batch_images.png, viz_statistics.png
```

**Test video results (setelah run inference.py):**
```bash
python inference.py video/2022111801.mp4
python visualize_video.py
```

## File Visualization Scripts

| File | Fungsi |
|------|--------|
| `visualize_results.py` | Static visualizations (image + batch + stats) |
| `visualize_video.py` | Time-series video frame analysis |
| `viz_dashboard.py` | Interactive menu untuk semua opsi |
| `VISUALIZATION_GUIDE.py` | Panduan lengkap (ini file) |

## What Each Visualization Shows

### **viz_single_image.png**
- Panel 1: Original image
- Panel 2: Model prediction heatmap (0-1 probability)
- Panel 3: Binary mask (threshold 0.5)
- Panel 4: Overlay on original (cyan = water)

### **viz_batch_images.png**
- 6 sample images dari satu folder
- 3 columns: Original | Prediction | Overlay
- Label: water percentage & status

### **viz_statistics.png**
- Histogram: Water % distribution
- Bar chart: Mean water % per folder
- Pie chart: Aman/Siaga/Waspada/Bahaya status %
- Table: Statistics (mean, std, min, max)

### **viz_video_*.png** (setelah test video)
- Time series: Frame-by-frame water detection
- Histogram: Distribution across all frames
- Pie chart: Status distribution
- Table: Video statistics

## Flood Status Colors

| Status | Color | Water % | Meaning |
|--------|-------|---------|---------|
| **Aman** | 🟢 Green | < 5% | Safe |
| **Siaga** | 🟡 Yellow | 5-15% | Alert |
| **Waspada** | 🟠 Orange | 15-30% | Warning |
| **Bahaya** | 🔴 Red | > 30% | Danger |

## Model Performance Summary

- **Validation IoU**: 0.9408 (94.08%)
- **Best Epoch**: 17/20
- **Tested on**: 20 videos, 12,981 frames
- **Average water detection**: 29.95%

---

**NEXT STEP:**
```bash
# 1. Lihat PNG files di folder
# 2. Atau jalankan: python viz_dashboard.py
# 3. Untuk video results: python inference.py video/[nama].mp4 → python visualize_video.py
```

**STATUS: ✅ VISUALIZATION READY FOR USE**
