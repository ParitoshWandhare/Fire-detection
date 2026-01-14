"""
Phase 2 - Create temporal sequences for ConvLSTM fire spread prediction

Goal:
Take 5 consecutive days → predict the next day's fire mask

Only sequences where:
- the same tile position exists on ALL 6 days (input + target)
- there is at least some fire somewhere in the window
are saved.
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re

# ===================== CONFIGURATION =====================
TILE_SIZE = 256                     # REQUIRED output size
SEQUENCE_LENGTH = 5                 # Number of input days
PREDICTION_HORIZON = 1              # Predict next 1 day
MIN_FIRE_PIXELS_IN_WINDOW = 1       # Skip empty windows

INPUT_DIR_IMAGES = Path("data/interim/tiles/images")
INPUT_DIR_MASKS  = Path("data/interim/tiles/masks")
OUTPUT_BASE_DIR  = Path("data/processed/sequences")

# Create output folders
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
for split in ["train", "val", "test"]:
    (OUTPUT_BASE_DIR / split).mkdir(exist_ok=True)

# ===================== HELPERS =====================
def extract_date_from_filename(filename: str) -> str | None:
    """Extract YYYY-MM-DD from filename"""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    return match.group(1) if match else None


def get_all_tiles_by_date():
    """Group all image + mask tiles by date"""
    tiles_by_date = {}

    print("Scanning image tiles...")
    for img_path in INPUT_DIR_IMAGES.glob("*.png"):
        date = extract_date_from_filename(img_path.name)
        if not date:
            continue

        mask_path = INPUT_DIR_MASKS / img_path.name
        if not mask_path.exists():
            print(f"Mask missing for {img_path.name}")
            continue

        tiles_by_date.setdefault(date, []).append((img_path, mask_path))

    sorted_dates = sorted(tiles_by_date.keys())

    print(f"\nFound {len(sorted_dates)} unique dates")
    print(f"Total tiles: {sum(len(v) for v in tiles_by_date.values())}")
    if sorted_dates:
        print("Date range:", sorted_dates[0], "→", sorted_dates[-1])

    return tiles_by_date, sorted_dates


# ===================== MAIN =====================
def create_sequences():
    tiles_by_date, sorted_dates = get_all_tiles_by_date()

    if len(sorted_dates) < SEQUENCE_LENGTH + PREDICTION_HORIZON:
        print("ERROR: Not enough dates for sequence creation")
        return

    sequence_id = 0
    print("\nCreating sliding window sequences...")

    for start_idx in tqdm(range(len(sorted_dates) - SEQUENCE_LENGTH)):
        input_dates = sorted_dates[start_idx : start_idx + SEQUENCE_LENGTH]
        target_date = sorted_dates[start_idx + SEQUENCE_LENGTH]
        window_dates = input_dates + [target_date]

        ref_tiles = tiles_by_date[input_dates[0]]

        for tile_idx in range(len(ref_tiles)):
            all_paths = []
            valid = True

            # Check tile exists on all days
            for day in window_dates:
                if tile_idx >= len(tiles_by_date.get(day, [])):
                    valid = False
                    break
                all_paths.append(tiles_by_date[day][tile_idx])

            if not valid:
                continue

            input_sequence = []
            total_fire_pixels = 0

            # Load input frames
            for day_idx in range(SEQUENCE_LENGTH):
                img_path, mask_path = all_paths[day_idx]

                # ---- RGB IMAGE ----
                img = cv2.imread(str(img_path))
                if img is None:
                    valid = False
                    break
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_AREA)
                img = img.astype(np.float16) / 255.0

                # ---- CURRENT MASK ----
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    valid = False
                    break
                mask = cv2.resize(mask, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(np.float16) / 255.0
                mask = mask[..., None]
                total_fire_pixels += np.sum(mask > 0)

                # ---- PREVIOUS MASK ----
                if day_idx == 0:
                    prev_mask = np.zeros_like(mask)
                else:
                    prev_path = all_paths[day_idx - 1][1]
                    prev_img = cv2.imread(str(prev_path), cv2.IMREAD_GRAYSCALE)
                    if prev_img is None:
                        valid = False
                        break
                    prev_img = cv2.resize(prev_img, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_NEAREST)
                    prev_mask = prev_img.astype(np.float16) / 255.0
                    prev_mask = prev_mask[..., None]

                frame = np.concatenate([img, prev_mask], axis=-1)  # [256,256,4]
                input_sequence.append(frame)

            if not valid or total_fire_pixels < MIN_FIRE_PIXELS_IN_WINDOW:
                continue

            # ---- TARGET MASK ----
            target_path = all_paths[-1][1]
            target_img = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
            if target_img is None:
                continue
            target_img = cv2.resize(target_img, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_NEAREST)
            target_mask = target_img.astype(np.float16) / 255.0
            target_mask = target_mask[..., None]

            # ---- STACK & SAVE (NO RAM BUILDUP) ----
            input_seq_array = np.stack(input_sequence).astype(np.float16)

            # Simple chronological split
            if start_idx < int(0.7 * len(sorted_dates)):
                split = "train"
            elif start_idx < int(0.9 * len(sorted_dates)):
                split = "val"
            else:
                split = "test"

            np.savez_compressed(
                OUTPUT_BASE_DIR / split / f"seq_{sequence_id:06d}.npz",
                input_sequence=input_seq_array,
                target_mask=target_mask
            )

            sequence_id += 1

            # FREE MEMORY
            del input_seq_array, target_mask, input_sequence

    print(f"\nDone! Total sequences saved: {sequence_id}")
    print("Check: data/processed/sequences/")

# ===================== ENTRY =====================
if __name__ == "__main__":
    create_sequences()
