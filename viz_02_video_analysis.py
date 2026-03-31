"""
VIDEO VISUALIZATION - Create analysis plots from video results
Show frame-by-frame water detection trends
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


def visualize_video_results(json_file):
    """
    Create visualizations from video result JSON
    """
    
    json_path = Path(json_file)
    if not json_path.exists():
        print(f"ERROR: {json_file} not found!")
        print("\nFirst run: python inference.py video/2022111801.mp4")
        return
    
    with open(json_path) as f:
        results = json.load(f)
    
    # Extract data
    video_name = results.get('video', 'Unknown')
    frames_data = results.get('frames', [])
    
    if not frames_data:
        print(f"No frame data in {json_file}")
        return
    
    frame_nums = [f['frame'] for f in frames_data]
    water_pcts = [f['water_percentage'] for f in frames_data]
    statuses = [f['status'] for f in frames_data]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Video Analysis: {Path(video_name).name}', fontsize=14, fontweight='bold')
    
    # [0,0] Time series water percentage
    status_colors = {
        'Aman': 'green',
        'Siaga': 'yellow',
        'Waspada': 'orange',
        'Bahaya': 'red'
    }
    
    colors = [status_colors.get(s, 'gray') for s in statuses]
    
    axes[0, 0].scatter(frame_nums, water_pcts, c=colors, alpha=0.6, s=30)
    axes[0, 0].plot(frame_nums, water_pcts, alpha=0.3, color='steelblue')
    axes[0, 0].axhline(5, color='green', linestyle='--', alpha=0.5, label='Aman threshold')
    axes[0, 0].axhline(15, color='yellow', linestyle='--', alpha=0.5, label='Siaga threshold')
    axes[0, 0].axhline(30, color='orange', linestyle='--', alpha=0.5, label='Waspada threshold')
    axes[0, 0].set_xlabel('Frame Number')
    axes[0, 0].set_ylabel('Water Percentage (%)')
    axes[0, 0].set_title('Frame-by-Frame Water Detection')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # [0,1] Histogram
    axes[0, 1].hist(water_pcts, bins=30, color='steelblue', edgecolor='black')
    axes[0, 1].axvline(np.mean(water_pcts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(water_pcts):.1f}%')
    axes[0, 1].axvline(np.median(water_pcts), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(water_pcts):.1f}%')
    axes[0, 1].set_xlabel('Water Percentage (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Water Detection')
    axes[0, 1].legend()
    
    # [1,0] Status distribution
    status_counts = defaultdict(int)
    for status in statuses:
        status_counts[status] += 1
    
    status_order = ['Aman', 'Siaga', 'Waspada', 'Bahaya']
    labels = [s for s in status_order if s in status_counts]
    sizes = [status_counts[s] for s in labels]
    colors_pie = [status_colors[s] for s in labels]
    
    axes[1, 0].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Flood Status Distribution')
    
    # [1,1] Statistics text
    axes[1, 1].axis('off')
    
    stats_text = f"""
VIDEO STATISTICS

Name: {Path(video_name).name}
Total Frames: {len(frames_data)}

Water Percentage:
  Mean: {np.mean(water_pcts):.2f}%
  Median: {np.median(water_pcts):.2f}%
  Std Dev: {np.std(water_pcts):.2f}%
  Min: {np.min(water_pcts):.2f}%
  Max: {np.max(water_pcts):.2f}%

Flood Status Counts:
  Aman: {status_counts.get('Aman', 0)} frames ({status_counts.get('Aman', 0)/len(frames_data)*100:.1f}%)
  Siaga: {status_counts.get('Siaga', 0)} frames ({status_counts.get('Siaga', 0)/len(frames_data)*100:.1f}%)
  Waspada: {status_counts.get('Waspada', 0)} frames ({status_counts.get('Waspada', 0)/len(frames_data)*100:.1f}%)
  Bahaya: {status_counts.get('Bahaya', 0)} frames ({status_counts.get('Bahaya', 0)/len(frames_data)*100:.1f}%)

Overall Status: {max(status_counts, key=status_counts.get).upper()}
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    return fig


def main():
    print("\n" + "="*80)
    print("VIDEO RESULTS VISUALIZATION")
    print("="*80 + "\n")
    
    # Try to find JSON files
    json_files = list(Path('video').glob('*_water_results.json'))
    
    if not json_files:
        print("ERROR: No video results found!")
        print("\nFirst, test a video:")
        print("  python inference.py video/2022111801.mp4")
        print("\nThen run this script again")
        return
    
    print(f"Found {len(json_files)} result files\n")
    
    for idx, json_file in enumerate(json_files[:3], 1):
        print(f"{idx}. Processing {json_file.name}...")
        
        fig = visualize_video_results(json_file)
        if fig:
            output_file = f"viz_video_{idx}.png"
            fig.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"   ✓ Saved: {output_file}\n")
    
    print("="*80)
    print("Video visualization complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
