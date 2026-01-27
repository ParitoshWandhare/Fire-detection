# scripts/visualize_convlstm_predictions.py
"""
Visualize multiple ConvLSTM predictions vs ground truth.
Shows 4 images per example:
- Last input mask (Day 5 previous mask)
- Model prediction (probability 0–1)
- Model binary prediction (>0.5 threshold)
- Ground truth (real Day 6 mask)

Run:
python -m scripts.visualize_convlstm_predictions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

from src.models.convlstm import FireSpreadPredictor

# ===================== CONFIG =====================
CHECKPOINT_PATH = Path("checkpoints/best_convlstm.pth")
VAL_DIR         = Path("data/processed/sequences/val")
NUM_EXAMPLES    = 6
THRESHOLD       = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")

# ===================== LOAD MODEL =====================
def load_model():
    if not CHECKPOINT_PATH.exists():
        print(f"Error: No model found at {CHECKPOINT_PATH}")
        return None
    
    model = FireSpreadPredictor(
        input_channels=4,
        hidden_channels=64
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    )
    model.eval()

    print(f"Model loaded successfully from: {CHECKPOINT_PATH}")
    return model

# ===================== VISUALIZE ONE EXAMPLE =====================
def visualize_one_example(npz_path, model, example_num):
    print(f"\nLoading example {example_num}: {npz_path.name}")
    
    data = np.load(npz_path)

    # ---------------- INPUT ----------------
    # input_sequence: [T, H, W, C] → [1, T, C, H, W]
    input_seq = torch.from_numpy(data["input_sequence"])
    input_seq = input_seq.permute(0, 3, 1, 2).unsqueeze(0)
    input_seq = input_seq.float().to(DEVICE)

    # ---------------- TARGET ----------------
    # target_mask: [H, W, 1] → [1, 1, H, W]
    target = torch.from_numpy(data["target_mask"])
    target = target.permute(2, 0, 1).unsqueeze(0)
    target = target.float().to(DEVICE)

    # ---------------- PREDICTION ----------------
    with torch.no_grad():
        pred = model(input_seq)  # [1, 1, H, W]

    # ---------------- TO NUMPY ----------------
    last_input_mask = input_seq[0, -1, 3].cpu().numpy()   # [H, W]
    pred_prob       = pred[0, 0].cpu().numpy()            # [H, W]
    pred_binary     = (pred_prob > THRESHOLD).astype(float)
    true_mask       = target[0, 0].cpu().numpy()          # ✅ [H, W]

    # ---------------- PLOT ----------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(
        f"Example {example_num} — {npz_path.stem}",
        fontsize=16,
        y=1.05
    )

    axes[0].imshow(last_input_mask, cmap="gray")
    axes[0].set_title("Last Input Mask (Day 5)")
    axes[0].axis("off")

    axes[1].imshow(pred_prob, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Prediction (Probability)")
    axes[1].axis("off")

    axes[2].imshow(pred_binary, cmap="gray")
    axes[2].set_title(f"Prediction > {THRESHOLD}")
    axes[2].axis("off")

    axes[3].imshow(true_mask, cmap="gray")
    axes[3].set_title("Ground Truth (Day 6)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show(block=True)

    print(f"Finished example {example_num}")

# ===================== MAIN =====================
def main():
    model = load_model()
    if model is None:
        return

    npz_files = list(VAL_DIR.glob("*.npz"))
    print(f"Found {len(npz_files)} validation sequences")

    if not npz_files:
        print("No validation files found.")
        return

    selected_files = random.sample(
        npz_files,
        min(NUM_EXAMPLES, len(npz_files))
    )

    print(f"Showing {len(selected_files)} random examples...\n")

    for i, npz_path in enumerate(selected_files, 1):
        visualize_one_example(npz_path, model, i)

if __name__ == "__main__":
    main()
