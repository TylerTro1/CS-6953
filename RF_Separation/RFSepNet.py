import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv1d(nn.Module):
    """
    Implements complex convolution using real-valued ops.
    Input: (B, 2C, T)
    Output: (B, 2C_out, T)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.real = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.imag = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        xr, xi = x.chunk(2, dim=1)

        real = self.real(xr) - self.imag(xi)
        imag = self.real(xi) + self.imag(xr)

        return torch.cat([real, imag], dim=1)
    

class RFEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()

        stride = 2 if downsample else 1

        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)
    


class RFTransformer(nn.Module):
    def __init__(self, dim, depth=6, heads=8, mlp_dim=1024):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        # x: (B, C, T)
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = self.transformer(x)
        return x.permute(0, 2, 1)
    


class RFDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)
    

class RFSepNet(nn.Module):
    def __init__(self, num_sources=2, base_channels=64, depth=4):
        super().__init__()

        self.num_sources = num_sources

        # Encoder
        self.encoders = nn.ModuleList()
        ch = 2  # I/Q
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoders.append(RFEncoderBlock(ch, out_ch))
            ch = out_ch

        # Transformer bottleneck
        self.transformer = RFTransformer(dim=ch)

        # Decoder
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            out_ch = base_channels * (2 ** i)
            self.decoders.append(RFDecoderBlock(ch, out_ch))
            ch = out_ch

        # Output head
        self.head = nn.Conv1d(ch, num_sources * 2, kernel_size=1)

    def forward(self, x):
        skips = []

        # Encoder
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        # Transformer
        x = self.transformer(x)

        # Decoder
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x)
            x = x + skip  # skip connection

        out = self.head(x)

        B, C, T = out.shape
        out = out.view(B, self.num_sources, 2, T)

        return out