# scripts/train_convlstm.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import os

# Your imports
from src.models.convlstm import FireSpreadPredictor
from src.datasets.sequence_dataset import FireSequenceDataset

# Config
BATCH_SIZE = 4
EPOCHS = 10                # Let's do a few more now that it's working
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

TRAIN_DIR = Path("data/processed/sequences/train")
VAL_DIR   = Path("data/processed/sequences/val")

def main():
    print(f"Using device: {DEVICE}")

    train_ds = FireSequenceDataset(TRAIN_DIR)
    val_ds   = FireSequenceDataset(VAL_DIR)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,  num_workers=0)

    model = FireSpreadPredictor(input_channels=4, hidden_channels=64).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float('inf')
    print("Starting training...")

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_convlstm.pth")
            print("  → Saved new best model!")

    print("Training finished!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()