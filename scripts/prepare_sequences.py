# scripts/prepare_sequences.py
"""
Phase 2 - Create temporal sequences for ConvLSTM fire spread prediction
Goal: Take 5 consecutive days → predict the next day's fire mask
Only sequences where the same tile position exists on ALL 6 days (input + target)
and there is at least some fire somewhere in the window are saved.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re

# ===================== CONFIGURATION (easy to change) =====================
TILE_SIZE = 256                     # Should match your tiling size
SEQUENCE_LENGTH = 5                 # Number of input days
PREDICTION_HORIZON = 1              # Predict next 1 day
MIN_FIRE_PIXELS_IN_WINDOW = 1       # Skip if total fire pixels in whole window < this

INPUT_DIR_IMAGES = Path("data/interim/tiles/images")
INPUT_DIR_MASKS  = Path("data/interim/tiles/masks")
OUTPUT_BASE_DIR  = Path("data/processed/sequences")

# Create output folders
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
for split in ["train", "val", "test"]:
    (OUTPUT_BASE_DIR / split).mkdir(exist_ok=True)

# ===================== HELPERS =====================
def extract_date_from_filename(filename: str) -> str | None:
    """Extract YYYY-MM-DD from any part of the filename"""
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    return match.group(1) if match else None

def get_all_tiles_by_date():
    """Group all tile paths by date"""
    tiles_by_date = {}
    
    print("Scanning image tiles...")
    for img_path in INPUT_DIR_IMAGES.glob("*.png"):
        date = extract_date_from_filename(img_path.name)
        if not date:
            print(f"Skipping file without date: {img_path.name}")
            continue
            
        mask_path = INPUT_DIR_MASKS / img_path.name  # same name in masks folder
        
        if not mask_path.exists():
            print(f"Mask missing for {img_path.name}")
            continue
            
        if date not in tiles_by_date:
            tiles_by_date[date] = []
        tiles_by_date[date].append((img_path, mask_path))
    
    sorted_dates = sorted(tiles_by_date.keys())
    
    print(f"\nFound {len(sorted_dates)} unique dates")
    print(f"Total tiles: {sum(len(v) for v in tiles_by_date.values())}")
    if sorted_dates:
        print("Date range:", sorted_dates[0], "→", sorted_dates[-1])
    
    return tiles_by_date, sorted_dates

# ===================== MAIN SEQUENCE CREATION =====================
def create_sequences():
    tiles_by_date, sorted_dates = get_all_tiles_by_date()
    
    if len(sorted_dates) < SEQUENCE_LENGTH + PREDICTION_HORIZON:
        print(f"ERROR: Not enough dates! Need at least {SEQUENCE_LENGTH + PREDICTION_HORIZON}")
        return

    sequence_id = 0
    sequences = []

    print("\nCreating sliding window sequences of 5 days → predict next day")
    for start_idx in tqdm(range(len(sorted_dates) - SEQUENCE_LENGTH)):
        input_dates = sorted_dates[start_idx : start_idx + SEQUENCE_LENGTH]
        target_date = sorted_dates[start_idx + SEQUENCE_LENGTH]
        
        window_dates = input_dates + [target_date]

        # Reference tiles from first day
        ref_tiles = tiles_by_date[input_dates[0]]
        num_possible_tiles = len(ref_tiles)

        for tile_idx in range(num_possible_tiles):
            # Check if this tile_idx exists on EVERY day in the window
            all_paths = []
            valid = True
            for day in window_dates:
                if day not in tiles_by_date or tile_idx >= len(tiles_by_date[day]):
                    valid = False
                    break
                all_paths.append(tiles_by_date[day][tile_idx])

            if not valid:
                continue  # this tile position doesn't exist on all days → skip

            # Now load the 5 input frames
            input_sequence = []
            for day_idx in range(SEQUENCE_LENGTH):
                img_path, mask_path = all_paths[day_idx]

                # Load RGB image
                img = cv2.imread(str(img_path))
                if img is None:
                    valid = False
                    break
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0

                # Load current mask (grayscale → 0-1)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    valid = False
                    break
                mask = mask.astype(np.float32) / 255.0
                mask = mask[..., None]

                # Previous mask (zero for first day)
                if day_idx == 0:
                    prev_mask = np.zeros_like(mask)
                else:
                    prev_path = all_paths[day_idx - 1][1]
                    prev_img = cv2.imread(str(prev_path), cv2.IMREAD_GRAYSCALE)
                    if prev_img is None:
                        valid = False
                        break
                    prev_mask = prev_img.astype(np.float32) / 255.0
                    prev_mask = prev_mask[..., None]

                # Stack RGB + previous mask
                frame = np.concatenate([img, prev_mask], axis=-1)  # [H, W, 4]
                input_sequence.append(frame)

            if not valid:
                continue

            # Load target mask
            target_path = all_paths[-1][1]
            target_img = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
            if target_img is None:
                continue
            target_mask = target_img.astype(np.float32) / 255.0
            target_mask = target_mask[..., None]  # [H, W, 1]

            # Finally save the sequence
            input_seq_array = np.stack(input_sequence)  # [5, H, W, 4]

            sequences.append({
                "input_sequence": input_seq_array,
                "target_mask": target_mask,
                "sequence_id": sequence_id,
                "dates": window_dates,
                "tile_idx": tile_idx
            })
            sequence_id += 1

    print(f"\nTotal valid sequences created: {len(sequences)}")

    # Split into train/val/test (simple chronological split)
    if sequences:
        n = len(sequences)
        train_end = int(n * 0.7)
        val_end = int(n * 0.9)

        splits = {
            "train": sequences[:train_end],
            "val": sequences[train_end:val_end],
            "test": sequences[val_end:]
        }

        for split_name, seq_list in splits.items():
            print(f"Saving {len(seq_list)} sequences to {split_name}/")
            for seq in tqdm(seq_list, desc=split_name):
                np.savez_compressed(
                    OUTPUT_BASE_DIR / split_name / f"seq_{seq['sequence_id']:06d}.npz",
                    input_sequence=seq["input_sequence"],
                    target_mask=seq["target_mask"]
                )

    print("\nDone! Check data/processed/sequences/ folders")

if __name__ == "__main__":
    create_sequences()