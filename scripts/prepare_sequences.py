#!/usr/bin/env python3
"""
Phase 2 - Step 1: Temporal Sequence Preparation for ConvLSTM

This script prepares temporal sequences from Phase 1 outputs for fire spread prediction.

Input:
  - Satellite images organized by state and date
  - Fire masks generated from Phase 1 U-Net model
  - Metadata CSV with dates and geographic bounds

Output:
  - Temporal sequences: [T=5, H=512, W=512, C=4] (5 days × RGB+mask)
  - Target masks: [H=512, W=512, C=1] (day T+1 prediction target)
  - Train/val/test splits
  - Metadata JSON with sequence information

Usage:
    python scripts/prepare_sequences.py \
        --images-dir data/raw/forest_fire_dataset \
        --masks-dir data/interim/masks \
        --metadata-csv data/raw/WorldView_Metadata.csv \
        --output-dir data/processed/sequences \
        --sequence-length 5 \
        --prediction-horizon 1 \
        --train-days 1-20 \
        --val-days 16-25 \
        --test-days 21-30
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TemporalSequenceGenerator:
    """
    Generate temporal sequences for ConvLSTM training.
    """
    
    def __init__(self, 
                 images_dir: str,
                 masks_dir: str,
                 metadata_csv: str,
                 output_dir: str,
                 sequence_length: int = 5,
                 prediction_horizon: int = 1,
                 target_size: Tuple[int, int] = (512, 512)):
        """
        Initialize sequence generator.
        
        Args:
            images_dir: Base directory containing state subdirectories with images
            masks_dir: Directory containing generated fire masks
            metadata_csv: Path to CSV with image metadata
            output_dir: Where to save processed sequences
            sequence_length: Number of input time steps (default: 5 days)
            prediction_horizon: Days ahead to predict (default: 1 day)
            target_size: Resize images/masks to this size
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.metadata_csv = metadata_csv
        self.output_dir = Path(output_dir)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.target_size = target_size
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val', 'test']:
            (self.output_dir / split).mkdir(exist_ok=True)
        
        # Load metadata
        self.metadata_df = self._load_metadata()
        self.state_date_mapping = self._organize_by_state_date()
        
        logger.info(f"Initialized TemporalSequenceGenerator")
        logger.info(f"  Sequence length: {sequence_length} days")
        logger.info(f"  Prediction horizon: {prediction_horizon} day(s)")
        logger.info(f"  Target size: {target_size}")
        logger.info(f"  Total images in metadata: {len(self.metadata_df)}")
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load and parse metadata CSV."""
        logger.info(f"Loading metadata from {self.metadata_csv}")
        df = pd.read_csv(self.metadata_csv)
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by state and date
        df = df.sort_values(['State', 'Date']).reset_index(drop=True)
        
        logger.info(f"Loaded metadata for {len(df)} images")
        logger.info(f"States: {df['State'].unique().tolist()}")
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        return df
    
    def _organize_by_state_date(self) -> Dict[str, Dict[str, Dict]]:
        """
        Organize metadata by state and date.
        
        Returns:
            {state: {date_str: {filename, bounds, etc}}}
        """
        state_date_map = defaultdict(dict)
        
        for _, row in self.metadata_df.iterrows():
            state = row['State']
            date_str = row['Date'].strftime('%Y-%m-%d')
            
            state_date_map[state][date_str] = {
                'filename': row['File Name'],
                'date': row['Date'],
                'bounds': {
                    'top_right_lat': row['Top Right Latitude'],
                    'top_right_lon': row['Top Right Longitude'],
                    'bottom_left_lat': row['Bottom Left Latitude'],
                    'bottom_left_lon': row['Bottom Left Longitude']
                }
            }
        
        return state_date_map
    
    def _load_image(self, state: str, filename: str) -> Optional[np.ndarray]:
        """
        Load satellite image.
        
        Args:
            state: State/province name
            filename: Image filename (without extension)
            
        Returns:
            RGB image as numpy array [H, W, 3] or None if not found
        """
        # Try different extensions and paths
        possible_paths = [
            self.images_dir / state / f"{filename}.png",
            self.images_dir / state / f"{filename}.tif",
            self.images_dir / state / filename / f"{filename}.png",
        ]
        
        for img_path in possible_paths:
            if img_path.exists():
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize to target size
                    if img.shape[:2] != self.target_size:
                        img = cv2.resize(img, self.target_size)
                    
                    # Normalize to [0, 1]
                    img = img.astype(np.float32) / 255.0
                    
                    return img
                except Exception as e:
                    logger.warning(f"Error loading {img_path}: {e}")
                    continue
        
        logger.warning(f"Could not find image: {state}/{filename}")
        return None
    
    def _load_mask(self, filename: str) -> Optional[np.ndarray]:
        """
        Load fire mask.
        
        Args:
            filename: Image filename (mask will be filename_mask.png)
            
        Returns:
            Binary mask as numpy array [H, W] or None if not found
        """
        mask_name = f"{Path(filename).stem}_mask.png"
        mask_path = self.masks_dir / mask_name
        
        if not mask_path.exists():
            # Try without _mask suffix
            mask_path = self.masks_dir / f"{Path(filename).stem}.png"
        
        if mask_path.exists():
            try:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    return None
                
                # Resize to target size
                if mask.shape != self.target_size:
                    mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
                
                # Normalize to [0, 1]
                mask = (mask > 127).astype(np.float32)
                
                return mask
            except Exception as e:
                logger.warning(f"Error loading mask {mask_path}: {e}")
                return None
        
        logger.warning(f"Could not find mask: {mask_name}")
        return None
    
    def _create_sequence(self, 
                        state: str, 
                        start_date: datetime,
                        num_days: int) -> Optional[Tuple[np.ndarray, List[Dict]]]:
        """
        Create a temporal sequence starting from start_date.
        
        Args:
            state: State name
            start_date: First day of sequence
            num_days: Number of days to include
            
        Returns:
            Tuple of (sequence_array [T, H, W, 4], metadata_list)
            or None if any day is missing
        """
        sequence = []
        metadata = []
        
        for day_offset in range(num_days):
            current_date = start_date + timedelta(days=day_offset)
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Check if this date exists for this state
            if date_str not in self.state_date_mapping[state]:
                logger.debug(f"Missing date {date_str} for state {state}")
                return None
            
            date_info = self.state_date_mapping[state][date_str]
            filename = date_info['filename']
            
            # Load image and mask
            image = self._load_image(state, filename)
            mask = self._load_mask(filename)
            
            if image is None or mask is None:
                logger.debug(f"Missing image or mask for {state}/{filename}")
                return None
            
            # Combine image (3 channels) + mask (1 channel) = 4 channels
            combined = np.concatenate([image, mask[..., np.newaxis]], axis=-1)
            sequence.append(combined)
            
            metadata.append({
                'date': date_str,
                'filename': filename,
                'bounds': date_info['bounds']
            })
        
        # Stack into [T, H, W, 4]
        sequence_array = np.stack(sequence, axis=0)
        
        return sequence_array, metadata
    
    def generate_sequences(self,
                          train_days: Tuple[int, int] = (1, 20),
                          val_days: Tuple[int, int] = (16, 25),
                          test_days: Tuple[int, int] = (21, 30)) -> Dict:
        """
        Generate all temporal sequences and split into train/val/test.
        
        Args:
            train_days: Day range for training (inclusive)
            val_days: Day range for validation (inclusive)
            test_days: Day range for testing (inclusive)
            
        Returns:
            Dictionary with statistics
        """
        logger.info("Starting sequence generation...")
        
        all_sequences = {
            'train': [],
            'val': [],
            'test': []
        }
        
        sequence_id = 0
        
        # Process each state
        for state in tqdm(self.state_date_mapping.keys(), desc="Processing states"):
            logger.info(f"\nProcessing state: {state}")
            
            # Get all dates for this state (sorted)
            dates = sorted([datetime.strptime(d, '%Y-%m-%d') 
                          for d in self.state_date_mapping[state].keys()])
            
            if len(dates) < self.sequence_length + self.prediction_horizon:
                logger.warning(f"State {state} has only {len(dates)} days, skipping")
                continue
            
            # Generate sequences using sliding window
            for i in range(len(dates) - self.sequence_length - self.prediction_horizon + 1):
                start_date = dates[i]
                
                # Determine which split this sequence belongs to
                day_of_month = start_date.day
                
                if train_days[0] <= day_of_month <= train_days[1]:
                    split = 'train'
                elif val_days[0] <= day_of_month <= val_days[1]:
                    split = 'val'
                elif test_days[0] <= day_of_month <= test_days[1]:
                    split = 'test'
                else:
                    continue  # Skip if not in any split
                
                # Create input sequence (T days)
                input_seq_result = self._create_sequence(
                    state, 
                    start_date, 
                    self.sequence_length
                )
                
                if input_seq_result is None:
                    continue
                
                input_sequence, input_metadata = input_seq_result
                
                # Create target (T + prediction_horizon day)
                target_date = start_date + timedelta(days=self.sequence_length)
                target_result = self._create_sequence(state, target_date, 1)
                
                if target_result is None:
                    continue
                
                target_array, target_metadata = target_result
                # Extract only the mask channel from target
                target_mask = target_array[0, :, :, 3]  # [H, W]
                
                # Create sequence info
                sequence_info = {
                    'sequence_id': f"seq_{sequence_id:06d}",
                    'state': state,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': target_date.strftime('%Y-%m-%d'),
                    'split': split,
                    'input_dates': [m['date'] for m in input_metadata],
                    'target_date': target_metadata[0]['date'],
                    'input_shape': list(input_sequence.shape),
                    'target_shape': list(target_mask.shape)
                }
                
                # Save sequence
                self._save_sequence(
                    sequence_id,
                    input_sequence,
                    target_mask,
                    sequence_info,
                    split
                )
                
                all_sequences[split].append(sequence_info)
                sequence_id += 1
        
        # Save metadata
        stats = self._save_metadata(all_sequences)
        
        return stats
    
    def _save_sequence(self,
                      sequence_id: int,
                      input_sequence: np.ndarray,
                      target_mask: np.ndarray,
                      sequence_info: Dict,
                      split: str):
        """
        Save a single sequence to disk.
        
        Args:
            sequence_id: Unique sequence identifier
            input_sequence: Input temporal sequence [T, H, W, 4]
            target_mask: Target fire mask [H, W]
            sequence_info: Metadata dictionary
            split: 'train', 'val', or 'test'
        """
        output_path = self.output_dir / split / f"sequence_{sequence_id:06d}.npz"
        
        # Save as compressed numpy array
        np.savez_compressed(
            output_path,
            input_sequence=input_sequence.astype(np.float32),
            target_mask=target_mask.astype(np.float32),
            metadata=json.dumps(sequence_info)
        )
    
    def _save_metadata(self, all_sequences: Dict) -> Dict:
        """
        Save metadata and statistics to JSON.
        
        Args:
            all_sequences: Dictionary with train/val/test sequence info
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'train_sequences': len(all_sequences['train']),
            'val_sequences': len(all_sequences['val']),
            'test_sequences': len(all_sequences['test']),
            'total_sequences': sum(len(v) for v in all_sequences.values()),
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'target_size': self.target_size,
            'states': list(self.state_date_mapping.keys()),
            'date_range': {
                'start': self.metadata_df['Date'].min().strftime('%Y-%m-%d'),
                'end': self.metadata_df['Date'].max().strftime('%Y-%m-%d')
            }
        }
        
        # Save complete metadata
        metadata = {
            'statistics': stats,
            'sequences': all_sequences,
            'generation_time': datetime.now().isoformat()
        }
        
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\nSequence generation complete!")
        logger.info(f"Statistics:")
        logger.info(f"  Training sequences: {stats['train_sequences']}")
        logger.info(f"  Validation sequences: {stats['val_sequences']}")
        logger.info(f"  Test sequences: {stats['test_sequences']}")
        logger.info(f"  Total sequences: {stats['total_sequences']}")
        logger.info(f"\nMetadata saved to: {metadata_path}")
        
        return stats
    
    def visualize_sample_sequence(self, sequence_id: int, split: str = 'train'):
        """
        Visualize a sample sequence for verification.
        
        Args:
            sequence_id: ID of sequence to visualize
            split: Which split to load from
        """
        import matplotlib.pyplot as plt
        
        sequence_path = self.output_dir / split / f"sequence_{sequence_id:06d}.npz"
        
        if not sequence_path.exists():
            logger.error(f"Sequence {sequence_id} not found in {split} split")
            return
        
        # Load sequence
        data = np.load(sequence_path)
        input_seq = data['input_sequence']  # [T, H, W, 4]
        target_mask = data['target_mask']    # [H, W]
        metadata = json.loads(str(data['metadata']))
        
        # Create visualization
        T = input_seq.shape[0]
        fig, axes = plt.subplots(2, T + 1, figsize=(4 * (T + 1), 8))
        
        for t in range(T):
            # Show RGB image
            axes[0, t].imshow(input_seq[t, :, :, :3])
            axes[0, t].set_title(f"Day {t+1}\n{metadata['input_dates'][t]}")
            axes[0, t].axis('off')
            
            # Show fire mask
            axes[1, t].imshow(input_seq[t, :, :, 3], cmap='hot')
            axes[1, t].set_title(f"Fire Mask Day {t+1}")
            axes[1, t].axis('off')
        
        # Show target
        axes[0, T].imshow(np.zeros_like(input_seq[0, :, :, :3]))
        axes[0, T].set_title(f"Target Day\n{metadata['target_date']}")
        axes[0, T].axis('off')
        
        axes[1, T].imshow(target_mask, cmap='hot')
        axes[1, T].set_title(f"Target Fire Mask")
        axes[1, T].axis('off')
        
        plt.suptitle(f"Sequence {sequence_id} - State: {metadata['state']}", 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_dir / f'sample_sequence_{sequence_id}.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {viz_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate temporal sequences for Phase 2 ConvLSTM training"
    )
    
    # Data paths
    parser.add_argument('--images-dir', type=str, required=True,
                       help='Directory with state subdirectories containing images')
    parser.add_argument('--masks-dir', type=str, required=True,
                       help='Directory containing Phase 1 generated masks')
    parser.add_argument('--metadata-csv', type=str, required=True,
                       help='Path to metadata CSV file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for processed sequences')
    
    # Sequence parameters
    parser.add_argument('--sequence-length', type=int, default=5,
                       help='Number of input time steps (default: 5 days)')
    parser.add_argument('--prediction-horizon', type=int, default=1,
                       help='Days ahead to predict (default: 1)')
    parser.add_argument('--target-size', type=int, nargs=2, default=[512, 512],
                       help='Target size for images/masks (default: 512 512)')
    
    # Data splits
    parser.add_argument('--train-days', type=str, default='1-20',
                       help='Day range for training (format: start-end)')
    parser.add_argument('--val-days', type=str, default='16-25',
                       help='Day range for validation (format: start-end)')
    parser.add_argument('--test-days', type=str, default='21-30',
                       help='Day range for testing (format: start-end)')
    
    # Visualization
    parser.add_argument('--visualize-samples', type=int, default=3,
                       help='Number of sample sequences to visualize')
    
    args = parser.parse_args()
    
    # Parse day ranges
    def parse_range(range_str):
        start, end = map(int, range_str.split('-'))
        return (start, end)
    
    train_days = parse_range(args.train_days)
    val_days = parse_range(args.val_days)
    test_days = parse_range(args.test_days)
    
    # Create generator
    generator = TemporalSequenceGenerator(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        target_size=tuple(args.target_size)
    )
    
    # Generate sequences
    stats = generator.generate_sequences(
        train_days=train_days,
        val_days=val_days,
        test_days=test_days
    )
    
    # Visualize sample sequences
    if args.visualize_samples > 0:
        logger.info(f"\nGenerating {args.visualize_samples} sample visualizations...")
        for i in range(min(args.visualize_samples, stats['train_sequences'])):
            generator.visualize_sample_sequence(i, split='train')
    
    logger.info("\n✅ Sequence preparation complete!")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info(f"📊 Total sequences: {stats['total_sequences']}")
    logger.info(f"   - Training: {stats['train_sequences']}")
    logger.info(f"   - Validation: {stats['val_sequences']}")
    logger.info(f"   - Testing: {stats['test_sequences']}")


if __name__ == '__main__':
    main()