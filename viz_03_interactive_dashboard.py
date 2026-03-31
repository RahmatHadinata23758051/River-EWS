"""
INTERACTIVE VISUALIZATION DASHBOARD
Menu utama untuk semua visualization options
"""

import subprocess
import sys
from pathlib import Path


def print_menu():
    print("\n" + "="*80)
    print("WATER DETECTION - VISUALIZATION DASHBOARD")
    print("="*80)
    print("""
[1] Visualize Single Image
    - Compare original + prediction + overlay
    
[2] Visualize Batch Images
    - 6 sample images dari satu folder
    
[3] Visualize Statistics
    - Water distribution histogram
    - Status distribution pie chart
    - Folder comparison
    
[4] Visualize Video Results
    - Frame-by-frame water percentage
    - Status distribution over time
    - Video statistics
    
[5] Run All Visualizations
    - Generate semua plot sekaligus
    
[6] Open Generated Images
    - Buka hasil visualization di folder
    
[0] Exit
    
""")
    print("="*80)


def run_visualization(option):
    """Run selected visualization"""
    
    if option == '1':
        print("\n[1] SINGLE IMAGE VISUALIZATION")
        print("-" * 80)
        image_path = input("Enter image path (default: images/2022111801/2022111801_000.jpg): ").strip()
        if not image_path:
            image_path = 'images/2022111801/2022111801_000.jpg'
        
        code = f"""
from visualize_results import Visualizer
viz = Visualizer()
fig = viz.visualize_comparison('{image_path}')
fig.savefig('viz_single.png', dpi=150, bbox_inches='tight')
print("✓ Saved: viz_single.png")
import matplotlib.pyplot as plt
plt.show()
"""
        exec(code)
    
    elif option == '2':
        print("\n[2] BATCH VISUALIZATION")
        print("-" * 80)
        folder = input("Enter folder path (default: images/2022111801): ").strip()
        if not folder:
            folder = 'images/2022111801'
        
        code = f"""
from visualize_results import Visualizer
viz = Visualizer()
fig = viz.visualize_batch('{folder}', num_samples=6)
fig.savefig('viz_batch.png', dpi=150, bbox_inches='tight')
print("✓ Saved: viz_batch.png")
import matplotlib.pyplot as plt
plt.show()
"""
        exec(code)
    
    elif option == '3':
        print("\n[3] STATISTICS VISUALIZATION")
        print("-" * 80)
        code = """
from visualize_results import Visualizer
viz = Visualizer()
folders = [
    'images/2022111801',
    'images/2022111802',
    'images/2023020101'
]
fig = viz.visualize_statistics(folders)
fig.savefig('viz_statistics.png', dpi=150, bbox_inches='tight')
print("✓ Saved: viz_statistics.png")
import matplotlib.pyplot as plt
plt.show()
"""
        exec(code)
    
    elif option == '4':
        print("\n[4] VIDEO RESULTS VISUALIZATION")
        print("-" * 80)
        json_file = input("Enter JSON file (default: video/2022111801_water_results.json): ").strip()
        if not json_file:
            json_file = 'video/2022111801_water_results.json'
        
        code = f"""
from visualize_video import visualize_video_results
fig = visualize_video_results('{json_file}')
if fig:
    fig.savefig('viz_video.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: viz_video.png")
    import matplotlib.pyplot as plt
    plt.show()
"""
        exec(code)
    
    elif option == '5':
        print("\n[5] RUNNING ALL VISUALIZATIONS")
        print("-" * 80)
        print("Generating all plots...")
        try:
            result = subprocess.run(
                [sys.executable, 'visualize_results.py'],
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
        except Exception as e:
            print(f"Error: {e}")
    
    elif option == '6':
        print("\n[6] OPENING RESULTS")
        print("-" * 80)
        import os
        os.system('explorer .')
        print("Folder opened in explorer")
    
    elif option == '0':
        print("\nExiting...")
        return False
    
    return True


def main():
    """Main interactive loop"""
    
    print("\n" + "="*80)
    print("WELCOME TO VISUALIZATION DASHBOARD")
    print("="*80)
    
    while True:
        print_menu()
        choice = input("Select option [0-6]: ").strip()
        
        if choice == '0':
            print("\n✓ Goodbye!\n")
            break
        elif choice in ['1', '2', '3', '4', '5', '6']:
            try:
                if not run_visualization(choice):
                    break
            except Exception as e:
                print(f"\n✗ Error: {e}\n")
        else:
            print("\n✗ Invalid option!\n")


if __name__ == "__main__":
    main()
