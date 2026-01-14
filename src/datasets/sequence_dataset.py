# src/datasets/sequence_dataset.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

class FireSequenceDataset(Dataset):
    """Loads .npz sequences for training ConvLSTM"""
    def __init__(self, sequences_dir):
        self.sequences_dir = Path(sequences_dir)
        self.files = sorted(list(self.sequences_dir.glob("*.npz")))
        print(f"Loaded {len(self.files)} sequences from {sequences_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        input_seq = data['input_sequence']      # [5, 256, 256, 4]
        target_mask = data['target_mask']       # [256, 256, 1]

        # Convert to torch tensors & add batch dimension later in DataLoader
        input_seq = torch.from_numpy(input_seq).permute(0, 3, 1, 2).float()  # [5, 4, 256, 256]
        target_mask = torch.from_numpy(target_mask).permute(2, 0, 1).float() # [1, 256, 256]

        return input_seq, target_mask

if __name__ == "__main__":
    # Test with train folder (change path if needed)
    dataset = FireSequenceDataset("data/processed/sequences/train")
    print("First sequence shapes:")
    inp, tgt = dataset[0]
    print("Input:", inp.shape)    # Should be torch.Size([5, 4, 256, 256])
    print("Target:", tgt.shape)   # torch.Size([1, 256, 256])