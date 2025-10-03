"""
Standalone debug script for fire detection validation.
No external project dependencies required - just numpy, opencv, matplotlib.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from typing import List, Tuple, Dict, Optional
import pandas as pd

print("Fire Detection Debug Tool - Standalone Version")
print("="*60)


class SimpleFireDetector:
    """Simple fire detection with configurable thresholds."""
    
    def __init__(self, config_path: str = "configs/data.yaml"):
        self.config = self.load_config(config_path)
        print(f"Loaded configuration:")
        print(f"  Red threshold: {self.config['red_threshold']}")
        print(f"  Orange ratio: {self.config['orange_ratio']}")
        print(f"  Brightness: {self.config['brightness_threshold']}")
        print(f"  Green max: {self.config['green_max']}")
        print(f"  Blue max: {self.config['blue_max']}")
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        defaults = {
            'red_threshold': 120,
            'orange_ratio': 0.6,
            'brightness_threshold': 200,
            'green_max': 160,
            'blue_max': 140,
            'red_green_contrast': 25,
            'red_blue_contrast': 30,
            'min_fire_area': 2,
            'noise_removal_enabled': True,
            'kernel_size': 3,
            'min_component_area': 1
        }
        
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            print("Using default parameters")
            return defaults
        
        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            fire_cfg = yaml_config.get('data', {}).get('fire_detection', {})
            thresholds = fire_cfg.get('rgb_thresholds', {})
            noise_cfg = fire_cfg.get('noise_removal', {})
            
            config = {
                'red_threshold': thresholds.get('red_threshold', defaults['red_threshold']),
                'orange_ratio': thresholds.get('orange_ratio', defaults['orange_ratio']),
                'brightness_threshold': thresholds.get('brightness_threshold', defaults['brightness_threshold']),
                'green_max': thresholds.get('green_max', defaults['green_max']),
                'blue_max': thresholds.get('blue_max', defaults['blue_max']),
                'red_green_contrast': thresholds.get('red_green_contrast', defaults['red_green_contrast']),
                'red_blue_contrast': thresholds.get('red_blue_contrast', defaults['red_blue_contrast']),
                'min_fire_area': thresholds.get('min_fire_area', defaults['min_fire_area']),
                'noise_removal_enabled': noise_cfg.get('enabled', defaults['noise_removal_enabled']),
                'kernel_size': noise_cfg.get('kernel_size', defaults['kernel_size']),
                'min_component_area': noise_cfg.get('min_component_area', defaults['min_component_area'])
            }
            
            return config
            
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default parameters")
            return defaults
    
    def detect(self, image: np.ndarray, 
               red_threshold: int = None,
               orange_ratio: float = None,
               brightness_threshold: int = None) -> np.ndarray:
        """
        Detect fire pixels in image.
        
        Args:
            image: RGB image array (H, W, 3)
            red_threshold: Override config value if provided
            orange_ratio: Override config value if provided
            brightness_threshold: Override config value if provided
            
        Returns:
            Boolean mask of fire pixels
        """
        # Use provided values or fall back to config
        red_thresh = red_threshold if red_threshold is not None else self.config['red_threshold']
        orange_ratio = orange_ratio if orange_ratio is not None else self.config['orange_ratio']
        bright_thresh = brightness_threshold if brightness_threshold is not None else self.config['brightness_threshold']
        
        red = image[:, :, 0].astype(float)
        green = image[:, :, 1].astype(float)
        blue = image[:, :, 2].astype(float)
        
        # Apply all fire detection conditions
        conditions = [
            red > red_thresh,
            red > green * orange_ratio,
            red > blue * 1.2,
            (red + green + blue) > bright_thresh,
            (red - green) > self.config['red_green_contrast'],
            (red - blue) > self.config['red_blue_contrast'],
            green < self.config['green_max'],
            blue < self.config['blue_max']
        ]
        
        mask = np.all(conditions, axis=0)
        
        # Apply noise removal
        if self.config['noise_removal_enabled']:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (self.config['kernel_size'], self.config['kernel_size'])
            )
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            
            # Remove small components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            cleaned_mask = np.zeros_like(mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= self.config['min_component_area']:
                    cleaned_mask[labels == i] = 1
            mask = cleaned_mask.astype(bool)
        
        return mask


def interactive_pixel_selection(image: np.ndarray) -> List[Tuple[int, int]]:
    """Allow user to click on fire pixels."""
    print("\n" + "="*60)
    print("INTERACTIVE PIXEL SELECTION")
    print("="*60)
    print("Instructions:")
    print("  1. Click on red/orange fire pixels in the image")
    print("  2. Selected points will be marked with red X")
    print("  3. Close the window when done")
    print("-"*60)
    
    coords = []
    
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            col, row = int(event.xdata), int(event.ydata)
            coords.append((row, col))
            print(f"  Selected pixel #{len(coords)}: (row={row}, col={col})")
            event.inaxes.plot(col, row, 'rx', markersize=12, markeredgewidth=2)
            event.canvas.draw()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(image)
    ax.set_title("Click on fire pixels (red/orange regions)\nClose window when done", 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()
    
    print(f"\nSelected {len(coords)} fire pixels")
    print("="*60)
    
    return coords


def analyze_fire_pixels(image: np.ndarray, 
                        coords: List[Tuple[int, int]], 
                        window_size: int = 20) -> Dict:
    """Analyze color statistics in fire regions."""
    if not coords:
        print("No coordinates provided for analysis")
        return {}
    
    fire_pixels = []
    
    for row, col in coords:
        # Extract window around pixel
        r_start = max(0, row - window_size//2)
        r_end = min(image.shape[0], row + window_size//2)
        c_start = max(0, col - window_size//2)
        c_end = min(image.shape[1], col + window_size//2)
        
        window = image[r_start:r_end, c_start:c_end]
        
        # Get bright red/orange pixels
        red_mask = window[:,:,0] > 150
        fire_pixels.extend(window[red_mask])
    
    if not fire_pixels:
        print("No fire pixels found in selected regions")
        return {}
    
    fire_pixels = np.array(fire_pixels)
    
    # Calculate statistics
    stats = {
        'n_pixels': len(fire_pixels),
        'red': {
            'mean': fire_pixels[:,0].mean(),
            'std': fire_pixels[:,0].std(),
            'min': fire_pixels[:,0].min(),
            'max': fire_pixels[:,0].max(),
            'median': np.median(fire_pixels[:,0])
        },
        'green': {
            'mean': fire_pixels[:,1].mean(),
            'std': fire_pixels[:,1].std(),
            'min': fire_pixels[:,1].min(),
            'max': fire_pixels[:,1].max(),
            'median': np.median(fire_pixels[:,1])
        },
        'blue': {
            'mean': fire_pixels[:,2].mean(),
            'std': fire_pixels[:,2].std(),
            'min': fire_pixels[:,2].min(),
            'max': fire_pixels[:,2].max(),
            'median': np.median(fire_pixels[:,2])
        }
    }
    
    # Calculate ratios
    red_green_ratio = fire_pixels[:,0] / (fire_pixels[:,1] + 1e-8)
    red_blue_ratio = fire_pixels[:,0] / (fire_pixels[:,2] + 1e-8)
    brightness = np.sum(fire_pixels, axis=1)
    red_green_diff = fire_pixels[:,0] - fire_pixels[:,1]
    red_blue_diff = fire_pixels[:,0] - fire_pixels[:,2]
    
    stats['ratios'] = {
        'red_green_ratio': {
            'mean': red_green_ratio.mean(),
            'min': red_green_ratio.min(),
            'max': red_green_ratio.max(),
            'median': np.median(red_green_ratio)
        },
        'red_blue_ratio': {
            'mean': red_blue_ratio.mean(),
            'min': red_blue_ratio.min(),
            'max': red_blue_ratio.max(),
            'median': np.median(red_blue_ratio)
        },
        'brightness': {
            'mean': brightness.mean(),
            'min': brightness.min(),
            'max': brightness.max(),
            'median': np.median(brightness)
        },
        'red_green_contrast': {
            'mean': red_green_diff.mean(),
            'min': red_green_diff.min(),
            'max': red_green_diff.max()
        },
        'red_blue_contrast': {
            'mean': red_blue_diff.mean(),
            'min': red_blue_diff.min(),
            'max': red_blue_diff.max()
        }
    }
    
    return stats


def print_statistics(stats: Dict):
    """Print detailed statistics."""
    if not stats:
        return
    
    print("\n" + "="*70)
    print(f"FIRE PIXEL STATISTICS (n={stats['n_pixels']} pixels)")
    print("="*70)
    
    print("\nCHANNEL VALUES:")
    print("-"*70)
    for channel in ['red', 'green', 'blue']:
        ch_stats = stats[channel]
        print(f"{channel.upper():6} | Mean: {ch_stats['mean']:6.1f} | "
              f"Std: {ch_stats['std']:5.1f} | "
              f"Min: {ch_stats['min']:3.0f} | "
              f"Max: {ch_stats['max']:3.0f} | "
              f"Median: {ch_stats['median']:6.1f}")
    
    print("\nRATIOS & CONTRASTS:")
    print("-"*70)
    ratios = stats['ratios']
    
    print(f"Red/Green Ratio    | Mean: {ratios['red_green_ratio']['mean']:.2f} | "
          f"Min: {ratios['red_green_ratio']['min']:.2f} | "
          f"Max: {ratios['red_green_ratio']['max']:.2f}")
    
    print(f"Red/Blue Ratio     | Mean: {ratios['red_blue_ratio']['mean']:.2f} | "
          f"Min: {ratios['red_blue_ratio']['min']:.2f} | "
          f"Max: {ratios['red_blue_ratio']['max']:.2f}")
    
    print(f"Brightness (R+G+B) | Mean: {ratios['brightness']['mean']:.0f} | "
          f"Min: {ratios['brightness']['min']:.0f} | "
          f"Max: {ratios['brightness']['max']:.0f}")
    
    print(f"Red-Green Contrast | Mean: {ratios['red_green_contrast']['mean']:.1f} | "
          f"Min: {ratios['red_green_contrast']['min']:.1f}")
    
    print(f"Red-Blue Contrast  | Mean: {ratios['red_blue_contrast']['mean']:.1f} | "
          f"Min: {ratios['red_blue_contrast']['min']:.1f}")
    
    print("\n" + "="*70)
    print("RECOMMENDED THRESHOLDS (Conservative - mean minus 1 std):")
    print("="*70)
    
    print(f"red_threshold:        {int(stats['red']['mean'] - stats['red']['std'])}")
    print(f"orange_ratio:         {ratios['red_green_ratio']['mean'] * 0.7:.2f}")
    print(f"brightness_threshold: {int(ratios['brightness']['mean'] - ratios['brightness']['std'])}")
    print(f"green_max:            {int(stats['green']['mean'] + stats['green']['std'])}")
    print(f"blue_max:             {int(stats['blue']['mean'] + stats['blue']['std'])}")
    print(f"red_green_contrast:   {int(ratios['red_green_contrast']['mean'] * 0.7)}")
    print(f"red_blue_contrast:    {int(ratios['red_blue_contrast']['mean'] * 0.7)}")
    
    print("\n" + "="*70)
    print("RECOMMENDED THRESHOLDS (Aggressive - mean minus 2 std):")
    print("="*70)
    
    print(f"red_threshold:        {int(stats['red']['mean'] - 2*stats['red']['std'])}")
    print(f"orange_ratio:         {ratios['red_green_ratio']['mean'] * 0.5:.2f}")
    print(f"brightness_threshold: {int(ratios['brightness']['mean'] - 2*ratios['brightness']['std'])}")
    print(f"green_max:            {int(stats['green']['mean'] + 2*stats['green']['std'])}")
    print(f"blue_max:             {int(stats['blue']['mean'] + 2*stats['blue']['std'])}")
    
    print("="*70 + "\n")


def visualize_detection(image: np.ndarray, mask: np.ndarray, save_path: str = None):
    """Visualize fire detection results."""
    fig = plt.figure(figsize=(18, 12))
    
    # Original image
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(image)
    ax1.set_title("Original Image", fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Fire mask
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(mask, cmap='hot')
    fire_count = np.sum(mask)
    fire_percent = (fire_count / mask.size) * 100
    ax2.set_title(f"Fire Mask\n{fire_count:,} pixels ({fire_percent:.3f}%)", 
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Overlay
    ax3 = plt.subplot(2, 3, 3)
    overlay = image.copy()
    overlay[mask] = [255, 0, 255]  # Magenta
    ax3.imshow(overlay)
    ax3.set_title("Detection Overlay", fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Red channel
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(image[:,:,0], cmap='Reds')
    ax4.set_title("Red Channel", fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    # Green channel
    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(image[:,:,1], cmap='Greens')
    ax5.set_title("Green Channel", fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # Blue channel
    ax6 = plt.subplot(2, 3, 6)
    ax6.imshow(image[:,:,2], cmap='Blues')
    ax6.set_title("Blue Channel", fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {save_path}")
    
    plt.show()


def parameter_sweep(detector: SimpleFireDetector, 
                   image: np.ndarray,
                   red_thresholds: List[int] = [150, 165, 180, 200],
                   orange_ratios: List[float] = [0.5, 0.7, 1.0, 1.3],
                   brightness_thresholds: List[int] = [200, 300, 400, 500]) -> pd.DataFrame:
    """Test multiple parameter combinations."""
    
    total = len(red_thresholds) * len(orange_ratios) * len(brightness_thresholds)
    print(f"\nRunning parameter sweep: {total} combinations...")
    
    results = []
    count = 0
    
    for red_thresh in red_thresholds:
        for orange_ratio in orange_ratios:
            for bright_thresh in brightness_thresholds:
                count += 1
                if count % 10 == 0:
                    print(f"  Progress: {count}/{total}")
                
                mask = detector.detect(image, red_thresh, orange_ratio, bright_thresh)
                fire_pixels = np.sum(mask)
                fire_percent = (fire_pixels / mask.size) * 100
                
                results.append({
                    'red_threshold': red_thresh,
                    'orange_ratio': orange_ratio,
                    'brightness_threshold': bright_thresh,
                    'fire_pixels': fire_pixels,
                    'fire_percent': fire_percent
                })
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("PARAMETER SWEEP RESULTS")
    print("="*70)
    print("\nTop 10 configurations (by fire pixel count):")
    print(df.nlargest(10, 'fire_pixels').to_string(index=False))
    print("\nBottom 10 configurations (by fire pixel count):")
    print(df.nsmallest(10, 'fire_pixels').to_string(index=False))
    print("="*70)
    
    return df


def visualize_sweep(sweep_df: pd.DataFrame):
    """Visualize parameter sweep results."""
    fig = plt.figure(figsize=(16, 10))
    
    # Fire pixels vs red threshold
    ax1 = plt.subplot(2, 2, 1)
    for orange_ratio in sorted(sweep_df['orange_ratio'].unique()):
        subset = sweep_df[sweep_df['orange_ratio'] == orange_ratio]
        ax1.plot(subset['red_threshold'], subset['fire_pixels'], 
                'o-', label=f'Orange={orange_ratio:.1f}', alpha=0.6, markersize=4)
    ax1.set_xlabel('Red Threshold', fontweight='bold')
    ax1.set_ylabel('Fire Pixels Detected', fontweight='bold')
    ax1.set_title('Effect of Red Threshold', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Fire pixels vs orange ratio
    ax2 = plt.subplot(2, 2, 2)
    for red_thresh in sorted(sweep_df['red_threshold'].unique()):
        subset = sweep_df[sweep_df['red_threshold'] == red_thresh]
        ax2.plot(subset['orange_ratio'], subset['fire_pixels'],
                'o-', label=f'Red={red_thresh}', alpha=0.6, markersize=4)
    ax2.set_xlabel('Orange Ratio', fontweight='bold')
    ax2.set_ylabel('Fire Pixels Detected', fontweight='bold')
    ax2.set_title('Effect of Orange Ratio', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Fire pixels vs brightness threshold
    ax3 = plt.subplot(2, 2, 3)
    for red_thresh in sorted(sweep_df['red_threshold'].unique()):
        subset = sweep_df[sweep_df['red_threshold'] == red_thresh]
        ax3.plot(subset['brightness_threshold'], subset['fire_pixels'],
                'o-', label=f'Red={red_thresh}', alpha=0.6, markersize=4)
    ax3.set_xlabel('Brightness Threshold', fontweight='bold')
    ax3.set_ylabel('Fire Pixels Detected', fontweight='bold')
    ax3.set_title('Effect of Brightness Threshold', fontweight='bold', fontsize=12)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Heatmap
    ax4 = plt.subplot(2, 2, 4)
    pivot = sweep_df.pivot_table(
        values='fire_pixels',
        index='red_threshold',
        columns='orange_ratio',
        aggfunc='mean'
    )
    im = ax4.imshow(pivot, cmap='hot', aspect='auto')
    ax4.set_xlabel('Orange Ratio Index', fontweight='bold')
    ax4.set_ylabel('Red Threshold', fontweight='bold')
    ax4.set_title('Fire Pixels Heatmap (avg over brightness)', fontweight='bold', fontsize=12)
    ax4.set_yticks(range(len(pivot.index)))
    ax4.set_yticklabels(pivot.index)
    plt.colorbar(im, ax=ax4, label='Fire Pixels')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("FIRE DETECTION VALIDATION TOOL - STANDALONE VERSION")
    print("="*70)
    
    # Configuration
    IMAGE_PATH = "data/interim/tiles/images/Ontario-2023-06-05_r002_c006.png"
    CONFIG_PATH = "configs/data.yaml"
    
    # Check image
    if not Path(IMAGE_PATH).exists():
        print(f"\nERROR: Image not found at: {IMAGE_PATH}")
        print("\nPlease update IMAGE_PATH in the script with your image location.")
        print("Example paths to try:")
        print("  - data/interim/tiles/images/")
        print("  - data/raw/forest_fire_dataset/")
        return
    
    # Load image
    print(f"\nLoading image: {IMAGE_PATH}")
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        print(f"ERROR: Failed to load image from {IMAGE_PATH}")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image loaded: {image.shape} (H x W x C)")
    
    # Initialize detector
    print(f"\nInitializing detector with config: {CONFIG_PATH}")
    detector = SimpleFireDetector(CONFIG_PATH)
    
    # Step 1: Interactive pixel selection
    print("\n" + "="*70)
    print("STEP 1: SELECT FIRE PIXELS")
    print("="*70)
    coords = interactive_pixel_selection(image)
    
    # Step 2: Analyze fire pixels
    if coords:
        print("\n" + "="*70)
        print("STEP 2: ANALYZE FIRE PIXEL STATISTICS")
        print("="*70)
        stats = analyze_fire_pixels(image, coords, window_size=20)
        print_statistics(stats)
    else:
        print("\nNo pixels selected. Skipping statistical analysis.")
        print("Proceeding with current configuration...")
    
    # Step 3: Test current configuration
    print("\n" + "="*70)
    print("STEP 3: TEST CURRENT CONFIGURATION")
    print("="*70)
    print("Detecting fire pixels with current settings...")
    mask = detector.detect(image)
    
    fire_count = np.sum(mask)
    fire_percent = (fire_count / mask.size) * 100
    print(f"\nDetection Results:")
    print(f"  Fire pixels detected: {fire_count:,}")
    print(f"  Percentage of image: {fire_percent:.3f}%")
    print(f"  Total image pixels: {mask.size:,}")
    
    # Step 4: Visualize results
    print("\n" + "="*70)
    print("STEP 4: VISUALIZE DETECTION RESULTS")
    print("="*70)
    visualize_detection(image, mask, save_path="debug_detection_results.png")
    
    # Step 5: Optional parameter sweep
    print("\n" + "="*70)
    print("STEP 5: PARAMETER SWEEP (OPTIONAL)")
    print("="*70)
    print("Run parameter sweep to find optimal thresholds?")
    print("This will test multiple parameter combinations.")
    print("Enter 'y' for yes, any other key to skip: ", end='')
    
    try:
        response = input().strip().lower()
    except:
        response = 'n'
    
    if response == 'y':
        print("\nRunning parameter sweep...")
        print("This may take 1-2 minutes depending on image size...")
        
        sweep_df = parameter_sweep(
            detector, 
            image,
            red_thresholds=[120, 160, 170, 185, 200],
            orange_ratios=[0.4, 0.6, 0.8, 1.0, 1.2],
            brightness_thresholds=[200, 280, 300, 400, 500]
        )
        
        # Save results
        output_path = "parameter_sweep_results.csv"
        sweep_df.to_csv(output_path, index=False)
        print(f"\nParameter sweep results saved to: {output_path}")
        
        # Visualize sweep results
        print("\nGenerating parameter sweep visualizations...")
        visualize_sweep(sweep_df)
        
        print("\nParameter sweep complete!")
    else:
        print("\nSkipping parameter sweep.")
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - debug_detection_results.png (visualization)")
    if response == 'y':
        print("  - parameter_sweep_results.csv (sweep data)")
    
    print("\nNext steps:")
    if coords and 'stats' in locals():
        print("  1. Review the recommended thresholds above")
        print("  2. Update configs/data.yaml with optimal values")
        print("  3. Re-run this script to validate changes")
    else:
        print("  1. Run this script again and select fire pixels")
        print("  2. Use statistics to optimize thresholds")
        print("  3. Update configs/data.yaml")
    
    if response == 'y':
        print("  4. Review parameter_sweep_results.csv for best parameters")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()