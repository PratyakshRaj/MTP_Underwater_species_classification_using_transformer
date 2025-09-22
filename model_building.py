import sys
sys.path.append("audioset_tagging_cnn/pytorch")

from models import Cnn6
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Torch hub is one thing
from torch.hub import load
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Position indices [0, 1, 2, ...]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        
        # Frequency terms for sine/cosine
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices; cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions
        
        # Add batch dimension → shape (1, max_len, d_model)
        pe = pe.unsqueeze(0)  
        
        # Register as buffer so it's saved with model but not a parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encodings added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class Cnn6FrameEmb(Cnn6):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # Run through convolutional feature extractor
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        
        # Instead of global pooling → return frame-level embeddings
        # Shape: (batch, channels, time, freq)
        return x

# Build model and load weights
def cnn_model():
    frame_model = Cnn6FrameEmb(sample_rate=32000, window_size=1024,
                            hop_size=320, mel_bins=64, fmin=50, fmax=14000,
                            classes_num=527)
    checkpoint = torch.load(r"C:\files_mtp\MTP\Cnn6_mAP=0.343.pth", map_location="cpu",
        weights_only=False)
    frame_model.load_state_dict(checkpoint['model'])
    return frame_model



class CNNTransformerModel(nn.Module):
    def __init__(self, mfcc_features=36,d_model=512, num_classes=55):
        super().__init__()
        self.cnn = cnn_model()
        for p in self.cnn.parameters():
            p.requires_grad = False
        for p in self.cnn.conv_block4.parameters():
            p.requires_grad = True

        self.mfcc_proj = nn.Linear(mfcc_features, d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, spec, mfcc,spec_mask=None,mfcc_mask=None):
        # (B, 1, time, mel)
        
        spec = spec.permute(0,1,3,2)
        cnn_out = self.cnn(spec)             # (B, C, T, F)
        cnn_out = cnn_out.mean(dim=3)        # avg over freq → (B, C, T)
        cnn_out = cnn_out.permute(0,2,1)     # (B, T, C)
       
        mfcc = mfcc.permute(0,2,1)           # (B, T, 36)
        mfcc_proj = self.mfcc_proj(mfcc)     # (B, T, d_model)

        
        if spec_mask is not None:
            # ensure mask matches transformer sequence length
            seq_len = cnn_out.shape[1]   # same as fused.shape[1]
            spec_mask_down = F.interpolate(
                spec_mask.float().unsqueeze(1),  # (B, 1, T)
                size=seq_len,
                mode="nearest"
            ).squeeze(1).bool()  # (B, seq_len)
        else:
            spec_mask_down = None
        
        
        fused, _ = self.cross_attn(query=cnn_out, key=mfcc_proj, value=mfcc_proj, key_padding_mask=~mfcc_mask if mfcc_mask is not None else None)
        fused = self.pos_enc(fused)
        encoded = self.transformer(fused,src_key_padding_mask=~spec_mask_down if spec_mask_down is not None else None)
        x = encoded.mean(dim=1)              # pooling over time
        out = self.classifier(x)
        return out

