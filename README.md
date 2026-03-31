# 🌊 Flood Detection Model (Computer Vision Module)

## 📌 Overview

This project implements a **semantic segmentation model for flood detection** using computer vision.

The model is designed as part of a larger **Early Warning System (EWS)**, where:

* 📡 Ultrasonic sensor → measures water level (objective data)
* 🎥 Computer vision → validates real-world flood conditions

> ⚠️ Current scope: **Computer Vision module only (validated & ready for integration)**
> 🔜 Next step: Sensor fusion (multi-modal validation)

---

## 🚀 Quick Start

### 1. Run Quick Test (Recommended)

```bash
python quick_test.py
```

Tests model on sample images and displays flood detection results.

---

### 2. Run Inference on Image

```bash
python inference.py images/2022111801/2022111801_000.jpg
```

---

### 3. Run Inference on Video

```bash
python inference.py videos/2022111801.mp4
```

---

## 🧠 Model Information

* **Architecture**: U-Net (Semantic Segmentation)
* **Task**: Binary segmentation (Water vs Non-water)
* **Validation IoU**: 94.08%
* **Test IoU**: ~89.39%
* **F1 Score**: 94.86%
* **Model Size**: 29.7 MB
* **Parameters**: 7.77M
* **Tested on**: 20 videos (12,981 frames)

---

## 🌊 Flood Status Classification

Water area percentage is used to determine flood status:

| Status  | Water Area (%) | Indicator  |
| ------- | -------------- | ---------- |
| Aman    | < 5%           | 🟢 Safe    |
| Siaga   | 5 – 15%        | 🟡 Alert   |
| Waspada | 15 – 30%       | 🟠 Warning |
| Bahaya  | > 30%          | 🔴 Danger  |

---

## 📊 Model Evaluation

Below is the evaluation of the model on 300 test samples:

![Model Evaluation](docs/model_evaluation.png)

### Key Metrics:

* **IoU**: 89.39% → High segmentation accuracy
* **Accuracy**: 96.52% → High pixel classification
* **Precision**: 94.35% → Low false positives
* **Recall**: 95.37% → Low missed detections
* **F1 Score**: 94.86% → Balanced performance

> ✅ The model demonstrates strong performance and stability for real-world flood detection scenarios.

---

## 📂 Dataset

* **3,574 images** (1280×720)
* **1,396 binary masks** (converted from multi-class annotations)
* **20 flood videos** for testing
* Source: Flood Amateur Video Dataset

### Preprocessing:

* Multi-class → Binary segmentation
* Water class extracted (Cyan color)
* Background merged

---

## 📁 Project Structure

```
flood_dataset/
├── quick_test.py              # Quick testing script (START HERE)
├── inference.py               # Image/video inference
├── 04_unet_model.py           # Model architecture
├── 05_train.py                # Training script
├── README.md
│
├── checkpoints/
│   └── best_model.pth
│
├── flood_detection_model/     # Production-ready module
│   ├── model/
│   ├── code/
│   └── setup.py
│
├── images/
├── videos/
├── annotations/
└── binary_masks/
```

---

## 📈 Testing Results

**Global Flood Status Distribution (20 videos, 12,981 frames):**

* 🔴 Bahaya: 47.8%
* 🟠 Waspada: 22.1%
* 🟡 Siaga: 15.8%
* 🟢 Aman: 14.3%

---

## 🔗 Integration (Next Phase)

The model is designed to be integrated with sensor data:

```python
if sensor_status == "Bahaya" and cv_detected:
    trigger_alarm()
```

> 🎯 Goal: Reduce false alarms using multi-modal validation

---

## ⚙️ Requirements

* Python 3.8+
* PyTorch 2.1+
* OpenCV
* NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

<img width="850" height="530" alt="image" src="https://github.com/user-attachments/assets/d5bac7c4-66b6-46b5-98ce-eca372d89b22" />
