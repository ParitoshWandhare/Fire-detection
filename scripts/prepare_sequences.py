# scripts/prepare_sequences.py
from pathlib import Path
import pandas as pd

# Paths (adjust if your folder names differ slightly)
RAW_IMAGES_DIR = Path("data/raw/forest_fire_dataset")
MASKS_DIR = Path("data/interim/masks")
METADATA_FILE = Path("data/raw/WorldView Metadata - Sheet1.csv")

def get_dates_per_province():
    provinces = [
        "Ontario",
        "Quebec" # match your folder name exactly
    ]
    
    dates_dict = {}
    
    for prov in provinces:
        prov_folder = RAW_IMAGES_DIR / prov
        if not prov_folder.exists():
            print(f"Warning: Folder not found - {prov_folder}")
            continue
            
        image_files = list(prov_folder.glob("*.*"))  # .tif or .png
        dates = []
        for f in image_files:
            # Extract date from filename, e.g. Ontario_2023-06-05.tif → 2023-06-05
            stem = f.stem
            if "_" in stem:
                date_part = stem.split("_")[-1]
                if len(date_part) == 10 and date_part.count("-") == 2:  # looks like YYYY-MM-DD
                    dates.append(date_part)
        
        dates = sorted(set(dates))  # remove duplicates if any
        dates_dict[prov] = dates
    
    return dates_dict

def print_summary(dates_dict):
    print("\n=== Date Summary per Province ===")
    for prov, dates in dates_dict.items():
        print(f"{prov:22} → {len(dates)} days  |  {dates[0]} → {dates[-1]}")
    print("Total provinces found:", len(dates_dict))

if __name__ == "__main__":
    dates_per_prov = get_dates_per_province()
    print_summary(dates_per_prov)
    
    # Optional: save to a text file for easy reference
    with open("date_summary.txt", "w") as f:
        for prov, dates in dates_per_prov.items():
            f.write(f"{prov}: {len(dates)} days ({dates[0]} to {dates[-1]})\n")