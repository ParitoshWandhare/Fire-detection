"""
<<<<<<< HEAD
Optimized U-Net model for forest fire detection - Target: F1 96%+, IoU 92%+

Key improvements:
1. Added SCSE (Spatial-Channel Squeeze & Excitation) attention
2. Improved skip connections with attention
3. Better feature fusion
4. Optimized for 4GB GPU (GTX 1650)
=======
Fixed U-Net model for forest fire detection.

Key fixes:
1. Corrected skip connection indexing in decoder
2. Added proper dropout and batch normalization
3. Fixed auxiliary output handling
4. Improved feature alignment
5. Added gradient flow improvements
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Optional, Tuple
import math


<<<<<<< HEAD
class SCSEModule(nn.Module):
    """Spatial and Channel Squeeze & Excitation attention module."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        
        # Channel attention
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        cse = self.cSE(x) * x
        # Spatial attention
        sse = self.sSE(x) * x
        # Combine
        return cse + sse


class ConvBlock(nn.Module):
    """Improved convolutional block with attention."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, padding: int = 1, dropout: float = 0.15,
                 use_attention: bool = False):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self.relu = nn.ReLU(inplace=True)
        
        # Optional attention
        self.attention = SCSEModule(out_channels) if use_attention else None
        
=======
class ConvBlock(nn.Module):
    """
    Improved convolutional block with proper normalization and dropout.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, padding: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self.relu = nn.ReLU(inplace=True)
        
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        if self.dropout and self.training:
            x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
<<<<<<< HEAD
        
        # Apply attention if enabled
        if self.attention:
            x = self.attention(x)
            
=======
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
        return x


class DecoderBlock(nn.Module):
<<<<<<< HEAD
    """Optimized decoder block with attention and improved skip connections."""
    
    def __init__(self, in_channels: int, skip_channels: int, 
                 out_channels: int, dropout: float = 0.15, use_attention: bool = True):
        super().__init__()
        
        # Upsampling
=======
    """
    Fixed decoder block with proper skip connection handling.
    """
    
    def __init__(self, in_channels: int, skip_channels: int, 
                 out_channels: int, dropout: float = 0.1):
        super().__init__()
        
        # Upsampling layer
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        
<<<<<<< HEAD
        # Skip connection adapter with attention
        self.skip_adapter = None
        self.skip_attention = None
        if skip_channels > 0 and skip_channels != in_channels // 2:
            self.skip_adapter = nn.Conv2d(skip_channels, in_channels // 2, 1, bias=False)
            if use_attention:
                self.skip_attention = SCSEModule(in_channels // 2)
        
        # Convolutional block
        conv_in_channels = in_channels // 2 + (in_channels // 2 if skip_channels > 0 else 0)
        self.conv_block = ConvBlock(conv_in_channels, out_channels, dropout=dropout, 
                                    use_attention=use_attention)
=======
        # Skip connection adapter if needed
        self.skip_adapter = None
        if skip_channels > 0 and skip_channels != in_channels // 2:
            self.skip_adapter = nn.Conv2d(skip_channels, in_channels // 2, 1, bias=False)
        
        # Convolutional block after concatenation
        conv_in_channels = in_channels // 2 + (in_channels // 2 if skip_channels > 0 else 0)
        self.conv_block = ConvBlock(conv_in_channels, out_channels, dropout=dropout)
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
        
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Upsample
        x = self.upsample(x)
        
        if skip is not None:
<<<<<<< HEAD
            # Ensure spatial match
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            
            # Adapt skip channels
            if self.skip_adapter is not None:
                skip = self.skip_adapter(skip)
                if self.skip_attention is not None:
                    skip = self.skip_attention(skip)
=======
            # Ensure spatial dimensions match
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            
            # Adapt skip channels if needed
            if self.skip_adapter is not None:
                skip = self.skip_adapter(skip)
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
            
            # Concatenate
            x = torch.cat([x, skip], dim=1)
        
        # Apply conv block
        x = self.conv_block(x)
        return x


class PretrainedEncoder(nn.Module):
<<<<<<< HEAD
    """Optimized encoder - using ResNet34 for 4GB GPU."""
    
    def __init__(self, backbone: str = "resnet34", pretrained: bool = True, 
                 in_channels: int = 3, dropout: float = 0.15):
=======
    """
    Improved pretrained encoder with better feature extraction.
    """
    
    def __init__(self, backbone: str = "resnet34", pretrained: bool = True, 
                 in_channels: int = 3, dropout: float = 0.1):
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
        super().__init__()
        
        self.backbone_name = backbone
        
<<<<<<< HEAD
        # Use ResNet34 - lighter than ResNet50, better for 4GB GPU
        if backbone == "resnet34":
            backbone_model = models.resnet34(pretrained=pretrained)
            self.channels = [64, 64, 128, 256, 512]
        elif backbone == "resnet18":
            backbone_model = models.resnet18(pretrained=pretrained)
            self.channels = [64, 64, 128, 256, 512]
        elif backbone == "resnet50":
            backbone_model = models.resnet50(pretrained=pretrained)
            self.channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # Extract layers
        self.conv1 = backbone_model.conv1
        self.bn1 = backbone_model.bn1
        self.relu = backbone_model.relu
        self.maxpool = backbone_model.maxpool
        
        self.layer1 = backbone_model.layer1
        self.layer2 = backbone_model.layer2
        self.layer3 = backbone_model.layer3
        self.layer4 = backbone_model.layer4
        
        # Modify first conv if needed
=======
        # Load pretrained backbone
        if backbone.startswith("resnet"):
            if backbone == "resnet18":
                backbone_model = models.resnet18(pretrained=pretrained)
                self.channels = [64, 64, 128, 256, 512]
            elif backbone == "resnet34":
                backbone_model = models.resnet34(pretrained=pretrained)
                self.channels = [64, 64, 128, 256, 512]
            elif backbone == "resnet50":
                backbone_model = models.resnet50(pretrained=pretrained)
                self.channels = [64, 256, 512, 1024, 2048]
            else:
                raise ValueError(f"Unsupported ResNet variant: {backbone}")
                
            # Extract encoder layers
            self.conv1 = backbone_model.conv1
            self.bn1 = backbone_model.bn1
            self.relu = backbone_model.relu
            self.maxpool = backbone_model.maxpool
            
            self.layer1 = backbone_model.layer1
            self.layer2 = backbone_model.layer2
            self.layer3 = backbone_model.layer3
            self.layer4 = backbone_model.layer4
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify first conv layer if input channels != 3
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
        if in_channels != 3:
            old_conv = self.conv1
            self.conv1 = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            
<<<<<<< HEAD
            if pretrained and in_channels > 3:
                with torch.no_grad():
                    self.conv1.weight[:, :3] = old_conv.weight
                    for i in range(3, in_channels):
                        self.conv1.weight[:, i:i+1] = old_conv.weight[:, i % 3:i % 3 + 1]
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        
        # Initial conv
=======
            # Initialize new conv layer
            if pretrained and in_channels > 3:
                with torch.no_grad():
                    # Copy first 3 channels from pretrained weights
                    self.conv1.weight[:, :3] = old_conv.weight
                    # Initialize additional channels
                    for i in range(3, in_channels):
                        self.conv1.weight[:, i:i+1] = old_conv.weight[:, i % 3:i % 3 + 1]
        
        # Add dropout to encoder if specified
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features in correct order for decoder.
        """
        features = []
        
        # Initial conv + pooling (C1: H/2, W/2)
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.dropout and self.training:
            x = self.dropout(x)
        c1 = x
        features.append(c1)
        
        x = self.maxpool(x)
        
        # ResNet layers
<<<<<<< HEAD
        c2 = self.layer1(x)
        features.append(c2)
        
        c3 = self.layer2(c2)
        features.append(c3)
        
        c4 = self.layer3(c3)
        features.append(c4)
        
        c5 = self.layer4(c4)
=======
        c2 = self.layer1(x)  # C2: H/4, W/4
        features.append(c2)
        
        c3 = self.layer2(c2)  # C3: H/8, W/8
        features.append(c3)
        
        c4 = self.layer3(c3)  # C4: H/16, W/16
        features.append(c4)
        
        c5 = self.layer4(c4)  # C5: H/32, W/32
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
        features.append(c5)
        
        return features


class UNet(nn.Module):
<<<<<<< HEAD
    """Optimized U-Net with attention - Target: F1 96%+, IoU 92%+"""
=======
    """
    Fixed U-Net model with proper skip connections and improved architecture.
    """
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
    
    def __init__(self, config: Dict):
        super().__init__()
        
<<<<<<< HEAD
        # Extract config
=======
        # Extract config parameters
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
        backbone = config.get("backbone", "resnet34")
        pretrained = config.get("pretrained", True)
        in_channels = config.get("in_channels", 3)
        self.num_classes = config.get("num_classes", 1)
        decoder_channels = config.get("decoder_channels", [256, 128, 64, 32])
<<<<<<< HEAD
        dropout = config.get("dropout", 0.15)
        self.aux_outputs = config.get("aux_outputs", True)
        
        # Attention config
        attention_config = config.get("attention", {})
        use_attention = attention_config.get("enabled", True)
        
        # Initialize encoder
        self.encoder = PretrainedEncoder(backbone, pretrained, in_channels, dropout)
        encoder_channels = self.encoder.channels
        
        # Initialize decoder with attention
        self.decoder_blocks = nn.ModuleList()
        prev_channels = encoder_channels[-1]
        
        for i, dec_channels in enumerate(decoder_channels):
            skip_idx = len(encoder_channels) - 2 - i
            skip_channels = encoder_channels[skip_idx] if skip_idx >= 0 else 0
=======
        dropout = config.get("dropout", 0.1)
        self.aux_outputs = config.get("aux_outputs", False)
        
        # Initialize encoder
        self.encoder = PretrainedEncoder(backbone, pretrained, in_channels, dropout)
        encoder_channels = self.encoder.channels  # [64, 64, 128, 256, 512]
        
        # Initialize decoder with fixed skip connections
        self.decoder_blocks = nn.ModuleList()
        
        # Start from deepest features (C5=512) and work up
        prev_channels = encoder_channels[-1]  # 512
        
        # Create decoder blocks: C5->C4, C4->C3, C3->C2, C2->C1
        for i, dec_channels in enumerate(decoder_channels):
            # Skip connection channels in reverse order
            skip_idx = len(encoder_channels) - 2 - i  # 3, 2, 1, 0 -> C4, C3, C2, C1
            
            if skip_idx >= 0:
                skip_channels = encoder_channels[skip_idx]
            else:
                skip_channels = 0
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
            
            decoder_block = DecoderBlock(
                in_channels=prev_channels,
                skip_channels=skip_channels,
                out_channels=dec_channels,
<<<<<<< HEAD
                dropout=dropout,
                use_attention=use_attention
=======
                dropout=dropout
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
            )
            
            self.decoder_blocks.append(decoder_block)
            prev_channels = dec_channels
        
<<<<<<< HEAD
        # Output head with intermediate bottleneck
        self.output_head = nn.Sequential(
            nn.Conv2d(prev_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, self.num_classes, kernel_size=1)
        )
        
        # Auxiliary classifiers
        if self.aux_outputs:
            self.aux_classifiers = nn.ModuleList([
                nn.Conv2d(channels, self.num_classes, kernel_size=1)
                for channels in decoder_channels[:-1]
            ])
        
        self._initialize_weights()
    
    def _initialize_weights(self):
=======
        # Final classifier head
        self.classifier = nn.Conv2d(prev_channels, self.num_classes, kernel_size=1)
        
        # Auxiliary classifiers for deep supervision
        if self.aux_outputs:
            self.aux_classifiers = nn.ModuleList([
                nn.Conv2d(channels, self.num_classes, kernel_size=1)
                for channels in decoder_channels[:-1]  # Skip the last one
            ])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights properly."""
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
<<<<<<< HEAD
        input_shape = x.shape[-2:]
        
        # Encoder
        encoder_features = self.encoder(x)
        
        # Decoder
        decoder_outputs = []
        x = encoder_features[-1]
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_idx = len(encoder_features) - 2 - i
=======
        """
        Fixed forward pass with proper skip connection alignment.
        """
        input_shape = x.shape[-2:]
        
        # Encoder forward pass
        encoder_features = self.encoder(x)  # [C1, C2, C3, C4, C5]
        
        # Decoder forward pass with correct skip connections
        decoder_outputs = []
        x = encoder_features[-1]  # Start with C5 (deepest features)
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Get corresponding skip features in correct order
            skip_idx = len(encoder_features) - 2 - i  # 3, 2, 1, 0 for C4, C3, C2, C1
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
            
            if skip_idx >= 0:
                skip_features = encoder_features[skip_idx]
                x = decoder_block(x, skip_features)
            else:
                x = decoder_block(x, None)
            
            decoder_outputs.append(x)
        
<<<<<<< HEAD
        # Final output
        out = self.output_head(x)
        
        # Resize if needed
=======
        # Final classification
        out = self.classifier(x)
        
        # Resize to input size if needed
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
        if out.shape[-2:] != input_shape:
            out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        
        results = {"out": out}
        
<<<<<<< HEAD
        # Auxiliary outputs
=======
        # Auxiliary outputs for deep supervision
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
        if self.aux_outputs and self.training:
            aux_outs = []
            for i, aux_classifier in enumerate(self.aux_classifiers):
                aux_out = aux_classifier(decoder_outputs[i])
                aux_out = F.interpolate(aux_out, size=input_shape, mode='bilinear', align_corners=False)
                aux_outs.append(aux_out)
            results["aux"] = aux_outs
        
        return results
    
<<<<<<< HEAD
    def predict(self, x: torch.Tensor, threshold: float = 0.45) -> torch.Tensor:
        """Inference with optimized threshold."""
=======
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Inference method with proper post-processing.
        """
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)["out"]
            if self.num_classes == 1:
<<<<<<< HEAD
                probs = torch.sigmoid(outputs)
                return (probs > threshold).float()
            else:
=======
                # Binary segmentation
                probs = torch.sigmoid(outputs)
                return (probs > threshold).float()
            else:
                # Multi-class segmentation
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
                return torch.softmax(outputs, dim=1).argmax(dim=1, keepdim=True).float()


def create_model(config: Dict) -> UNet:
<<<<<<< HEAD
    """Factory function."""
    return UNet(config)


if __name__ == "__main__":
=======
    """
    Factory function to create U-Net model from config.
    """
    return UNet(config)


# Test the fixed model
if __name__ == "__main__":
    # Test configuration
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
    config = {
        "backbone": "resnet34",
        "pretrained": True,
        "in_channels": 3,
        "num_classes": 1,
        "decoder_channels": [256, 128, 64, 32],
<<<<<<< HEAD
        "dropout": 0.15,
        "aux_outputs": True,
        "attention": {"enabled": True, "type": "scse"}
    }
    
    print("Testing optimized U-Net with attention:")
    model = create_model(config)
    
    x = torch.randn(2, 3, 512, 512)
    model.train()
    outputs = model(x)
    print(f"Main output: {outputs['out'].shape}")
    if "aux" in outputs:
        print(f"Auxiliary outputs: {len(outputs['aux'])}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("✅ Optimized U-Net ready!")
=======
        "dropout": 0.1,
        "aux_outputs": True
    }
    
    print("Testing fixed U-Net model:")
    model = create_model(config)
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    
    model.train()
    outputs = model(x)
    print(f"Main output shape: {outputs['out'].shape}")
    if "aux" in outputs:
        print(f"Auxiliary outputs: {len(outputs['aux'])}")
        for i, aux in enumerate(outputs["aux"]):
            print(f"  Aux {i}: {aux.shape}")
    
    # Test inference
    model.eval()
    pred = model.predict(x)
    print(f"Prediction shape: {pred.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("✅ Fixed U-Net model working correctly!")
>>>>>>> f28875eaafdd4bd2510d1c05f8e313882794caf2
