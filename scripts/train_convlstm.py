# scripts/train_convlstm.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm  # ← NEW: for progress bar

# Your imports
from src.models.convlstm import FireSpreadPredictor
from src.datasets.sequence_dataset import FireSequenceDataset

# ===================== CONFIG =====================
BATCH_SIZE = 4
EPOCHS = 20
LR = 0.00005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

TRAIN_DIR = Path("data/processed/sequences/train")
VAL_DIR   = Path("data/processed/sequences/val")

# ===================== COMBINED LOSS =====================
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
    
    def dice_loss(self, pred, target, smooth=1e-6):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice_loss(pred, target)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

def main():
    print(f"Using device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    train_ds = FireSequenceDataset(TRAIN_DIR)
    val_ds   = FireSequenceDataset(VAL_DIR)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = FireSpreadPredictor(input_channels=4, hidden_channels=64).to(DEVICE)
    criterion = CombinedLoss(alpha=0.5)  # BCE + Dice
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    print("Starting training...\n")

    for epoch in range(EPOCHS):
        # Training with progress bar
        model.train()
        train_loss = 0.0
        
        # NEW: tqdm progress bar for batches in this epoch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=True)
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar with current running loss
            avg_loss_so_far = train_loss / (batch_idx + 1)
            progress_bar.set_postfix(loss=f"{avg_loss_so_far:.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # Validation (no progress bar needed, it's faster)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} completed | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Best Val so far: {best_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_convlstm.pth")
            print("  → Saved new best model!")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

    print("\nTraining finished!")
    print(f"Final best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()