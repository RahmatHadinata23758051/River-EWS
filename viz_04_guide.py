"""
GUIDE - Cara Visualisasi Hasil Model
Complete visualization options untuk water detection model
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║        WATER DETECTION MODEL - VISUALIZATION GUIDE                        ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 VISUALISASI RESULTS - PANDUAN LENGKAP
══════════════════════════════════════════════════════════════════════════════

Saya udah generate 3 file PNG:
  ✓ viz_single_image.png     - Single image dengan 4 panel
  ✓ viz_batch_images.png     - 6 image samples side-by-side  
  ✓ viz_statistics.png       - Statistical analysis & distribution

══════════════════════════════════════════════════════════════════════════════

🎯 OPSI 1: SINGLE IMAGE VISUALIZATION
══════════════════════════════════════════════════════════════════════════════

COMMAND:
    python viz_dashboard.py → pilih [1]
    
ATAU langsung:
    python -c "from visualize_results import Visualizer; viz = Visualizer(); 
    fig = viz.visualize_comparison('images/2022111801/2022111801_000.jpg'); 
    fig.savefig('custom_viz.png'); import matplotlib.pyplot as plt; plt.show()"

OUTPUT:
    🖼️  viz_single.png (4-panel visualization):
    
    Panel 1 (Top-Left):     Original Image
    Panel 2 (Top-Right):    Model Prediction (Heatmap 0-1)
                            - Red = water (high probability)
                            - Blue = no water (low probability)
    Panel 3 (Bottom-Left):  Binary Mask (0.5 threshold)
                            - White = detected water
                            - Black = no water
    Panel 4 (Bottom-Right): Overlay on Original
                            - Cyan = predicted water
                            - Original = other areas

GUNAKAN UNTUK:
    ✓ Debug individual images
    ✓ Check prediction quality
    ✓ Understand model confidence

══════════════════════════════════════════════════════════════════════════════

🎯 OPSI 2: BATCH IMAGE VISUALIZATION
══════════════════════════════════════════════════════════════════════════════

COMMAND:
    python viz_dashboard.py → pilih [2]

OUTPUT:
    🖼️  viz_batch.png (6 images, 3 columns):
    
    Column 1: Original images dengan water % label
    Column 2: Model predictions (heatmap)
    Column 3: Overlay pada original
    
    Example output:
    Row 1: 2022111801_000.jpg  → 72.72% (Bahaya)
    Row 2: 2022111801_001.jpg  → 72.58% (Bahaya)
    Row 3: 2022111801_002.jpg  → 72.29% (Bahaya)
    ...

GUNAKAN UNTUK:
    ✓ Quick quality check
    ✓ Visual comparison antar images
    ✓ Verification across folder

══════════════════════════════════════════════════════════════════════════════

🎯 OPSI 3: STATISTICS VISUALIZATION  
══════════════════════════════════════════════════════════════════════════════

COMMAND:
    python viz_dashboard.py → pilih [3]

OUTPUT:
    🖼️  viz_statistics.png (4-panel):
    
    Panel 1 (Top-Left):     Histogram
                            - Distribution of water percentages
                            - Red line = Mean
    
    Panel 2 (Top-Right):    Bar Chart
                            - Mean water % per folder
                            - Compare different areas
    
    Panel 3 (Bottom-Left):  Pie Chart
                            - Flood status distribution
                            - Aman, Siaga, Waspada, Bahaya %
    
    Panel 4 (Bottom-Right): Statistics Table
                            - Total samples
                            - Mean, Std dev, Min, Max
    
    Example stats:
    Aman (Safe):    15.3% (green)
    Siaga (Alert):  18.2% (yellow)
    Waspada (Warn): 24.1% (orange)
    Bahaya (Danger):42.4% (red)

GUNAKAN UNTUK:
    ✓ Overall analysis
    ✓ Compare different regions
    ✓ Statistical reporting
    ✓ Data quality checks

══════════════════════════════════════════════════════════════════════════════

🎯 OPSI 4: VIDEO RESULTS VISUALIZATION
══════════════════════════════════════════════════════════════════════════════

STEP 1: Test video (generate JSON)
    python inference.py video/2022111801.mp4
    
    Generated:
    ✓ video/2022111801_water.mp4       (with overlay)
    ✓ video/2022111801_water_results.json (frame data)

STEP 2: Visualize results
    python viz_dashboard.py → pilih [4]

OUTPUT:
    🖼️  viz_video.png (4-panel):
    
    Panel 1 (Top-Left):     Time Series Plot
                            - X axis: Frame number
                            - Y axis: Water percentage
                            - Color: Status (Aman/Siaga/Waspada/Bahaya)
                            - Dashed lines: Thresholds
    
    Panel 2 (Top-Right):    Histogram
                            - Distribution of water detection
                            - Mean & median lines
    
    Panel 3 (Bottom-Left):  Pie Chart
                            - Status distribution across all frames
                            - How many Aman/Siaga/Waspada/Bahaya
    
    Panel 4 (Bottom-Right): Statistics
                            - Video name
                            - Total frames
                            - Mean/median/std/min/max water %
                            - Frame counts per status
                            - Overall classification

GUNAKAN UNTUK:
    ✓ Analyze trends over time
    ✓ Identify flooding patterns
    ✓ Generate reports

══════════════════════════════════════════════════════════════════════════════

🎯 OPSI 5: RUN ALL VISUALIZATIONS
══════════════════════════════════════════════════════════════════════════════

COMMAND:
    python viz_dashboard.py → pilih [5]

GENERATES:
    ✓ viz_single_image.png
    ✓ viz_batch_images.png
    ✓ viz_statistics.png

GUNAKAN UNTUK:
    ✓ Generate semua plots sekaligus
    ✓ Create report package

══════════════════════════════════════════════════════════════════════════════

📁 VISUALIZATION FILES LOCATION
══════════════════════════════════════════════════════════════════════════════

Semua file PNG tersimpan di folder project root:
    
    ./
    ├── viz_single_image.png
    ├── viz_batch_images.png
    ├── viz_statistics.png
    └── viz_*.png (hasil custom)

Untuk membuka semua file:
    python viz_dashboard.py → pilih [6]
    (akan buka explorer)

══════════════════════════════════════════════════════════════════════════════

🎨 COLOR CODING
══════════════════════════════════════════════════════════════════════════════

Heatmap Predictions:
    🔵 Blue   = Low probability (non-water)
    🟡 Yellow = Medium probability
    🔴 Red    = High probability (water)

Flood Status:
    🟢 Green  = Aman (Safe, <5% water)
    🟡 Yellow = Siaga (Alert, 5-15% water)
    🟠 Orange = Waspada (Warning, 15-30% water)
    🔴 Red    = Bahaya (Danger, >30% water)

══════════════════════════════════════════════════════════════════════════════

💡 TIPS & TRICKS
══════════════════════════════════════════════════════════════════════════════

1. Compare different folders:
   Modify visualize_results.py → visualize_statistics()
   Change folders list to: [
       'images/2022111801',  # Highly flooded
       'images/2023020101',  # Minimal water
       'images/2022111805'   # Mixed
   ]

2. Custom single image testing:
   python viz_dashboard.py → [1] → enter full path
   
3. Batch size adjustment:
   visualize_batch('folder', num_samples=10)  # 10 instead of 6

4. Save figures with better quality:
   fig.savefig('output.png', dpi=300)  # Higher DPI

5. Export statistics to JSON:
   Add json.dump(stats, open('stats.json', 'w'))

══════════════════════════════════════════════════════════════════════════════

✅ QUICK START VISUALIZATION WORKFLOW
══════════════════════════════════════════════════════════════════════════════

Step 1: Generate static visualizations
    python visualize_results.py
    → Creates: viz_single_image.png, viz_batch_images.png, viz_statistics.png

Step 2: Test video and visualize
    python inference.py video/2022111801.mp4
    python visualize_video.py
    → Creates: viz_video_1.png

Step 3: Use dashboard for custom visualization
    python viz_dashboard.py
    → Interactive menu for all options

Step 4: Open results
    python viz_dashboard.py → [6]
    → Opens explorer with all PNG files

══════════════════════════════════════════════════════════════════════════════

❓ FAQ
══════════════════════════════════════════════════════════════════════════════

Q: Gimana kalau mau visualisasi yang lain?
A: Edit visualize_results.py atau visualize_video.py, tambah logic custom

Q: Video visualization error?
A: Pastikan dulu run: python inference.py video/[nama].mp4
   Ini akan generate JSON yang dibutuhkan

Q: Gambar terlalu kecil?
A: Ubah figsize di script:
   fig, axes = plt.subplots(figsize=(20, 15))  # Lebih besar

Q: Mau export ke format lain?
A: Ubah format: fig.savefig('output.pdf')  atau .jpg, .tiff, dll

Q: Combine semua visualisasi jadi 1 file?
A: Bisa, tapi perlu custom script

══════════════════════════════════════════════════════════════════════════════

🎯 NEXT STEPS  
══════════════════════════════════════════════════════════════════════════════

Setelah visualization:
1. Check hasil PNG files di folder
2. Review prediction accuracy
3. Identify edge cases
4. Proceed ke sensor integration (07_ews_integration.py) jika diperlukan

══════════════════════════════════════════════════════════════════════════════

STATUS: ✅ VISUALIZATION TOOLS READY
Files generated: 3 PNG images
Ready for: Analysis & Reporting

""")
