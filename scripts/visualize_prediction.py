# scripts/visualize_prediction.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.convlstm import FireSpreadPredictor

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Load model
# ----------------------------
model = FireSpreadPredictor().to(device)
model.load_state_dict(
    torch.load("checkpoints/best_convlstm.pth", weights_only=True)
)
model.eval()

# ----------------------------
# Load validation sequence
# ----------------------------
data = np.load("data/processed/sequences/val/seq_000869.npz")

# input_sequence: (T, H, W, C)
input_seq = torch.from_numpy(data["input_sequence"]).float()
input_seq = input_seq.permute(0, 3, 1, 2)  # T,C,H,W
input_seq = input_seq.unsqueeze(0).to(device)  # 1,T,C,H,W

# target_mask: could be (H,W) or (1,H,W) or (1,1,H,W)
target_mask = torch.from_numpy(data["target_mask"]).float()

# ---- normalize target mask to (H, W)
if target_mask.ndim == 4:
    target_img = target_mask[0, 0]
elif target_mask.ndim == 3:
    target_img = target_mask[0]
else:
    target_img = target_mask

# ----------------------------
# Prediction
# ----------------------------
with torch.no_grad():
    pred = model(input_seq)  # (1, 1, H, W)

# ----------------------------
# Visualization
# ----------------------------
plt.figure(figsize=(12, 4))

# Last input fire mask (channel 3)
plt.subplot(1, 3, 1)
plt.imshow(input_seq[0, -1, 3].cpu(), cmap="hot")
plt.title("Last input mask")
plt.axis("off")

# Prediction
plt.subplot(1, 3, 2)
plt.imshow((pred[0, 0].cpu() > 0.5), cmap="hot")
plt.title("Prediction")
plt.axis("off")

# Ground truth
plt.subplot(1, 3, 3)
plt.imshow(target_img, cmap="hot")
plt.title("Ground truth")
plt.axis("off")

plt.tight_layout()
plt.show()
