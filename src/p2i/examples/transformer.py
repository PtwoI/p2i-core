import torch
from torch import nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
    def __init__(self, width=64, heads=4):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        # The fused operation remains fused in the IR when PyTorch exposes it so.
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(b, t, d))

class TransformerBlock(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = SelfAttention(width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class TinyTransformer(nn.Module):
    def __init__(self, width=64, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(width) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 8)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))
