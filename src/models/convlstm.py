# src/models/convlstm.py
"""
Simple ConvLSTM model for fire spread prediction
Input shape: [batch, time=5, channels=4 (RGB + prev_mask), height=256, width=256]
Output shape: [batch, 1, height=256, width=256] → predicted fire mask
"""

import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    """One ConvLSTM cell - processes one time step"""
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        # 4 gates: input, forget, cell, output
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
    
    def forward(self, input_tensor, hidden_state):
        h, c = hidden_state
        
        combined = torch.cat([input_tensor, h], dim=1)  # [batch, channels+hidden, H, W]
        gates = self.conv(combined)
        
        i, f, o, g = gates.chunk(4, dim=1)  # split into 4 parts
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class FireSpreadPredictor(nn.Module):
    """Complete model with 2 ConvLSTM layers + final prediction"""
    def __init__(self, input_channels=4, hidden_channels=64, kernel_size=3):
        super().__init__()
        
        self.cell1 = ConvLSTMCell(input_channels, hidden_channels, kernel_size)
        self.cell2 = ConvLSTMCell(hidden_channels, hidden_channels, kernel_size)
        
        # Final layers to predict fire mask
        self.conv_final = nn.Conv2d(hidden_channels, 1, kernel_size=1)
    
    def forward(self, x):
        """
        x shape: [batch, time=5, channels=4, H, W]
        Returns: predicted mask [batch, 1, H, W]
        """
        batch, time, channels, H, W = x.shape
        
        # Initialize hidden state with zeros
        h = torch.zeros(batch, self.cell1.hidden_dim, H, W).to(x.device)
        c = torch.zeros(batch, self.cell1.hidden_dim, H, W).to(x.device)
        
        # Process each time step
        for t in range(time):
            input_frame = x[:, t]  # [batch, 4, H, W]
            h, c = self.cell1(input_frame, (h, c))
            h, c = self.cell2(h, (h, c))  # second layer
        
        # Last hidden state → final prediction
        out = self.conv_final(h)  # [batch, 1, H, W]
        out = torch.sigmoid(out)  # make it 0–1 probability
        
        return out

# Quick test (run this file directly)
if __name__ == "__main__":
    model = FireSpreadPredictor()
    dummy_input = torch.randn(2, 5, 4, 256, 256)  # batch=2, time=5, channels=4
    output = model(dummy_input)
    print("Model output shape:", output.shape)  # Should be [2, 1, 256, 256]