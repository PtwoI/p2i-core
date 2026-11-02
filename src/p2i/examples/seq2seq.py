"""Small encoder-decoder Transformer with positions and causal decoding."""
import math
import torch
from torch import nn


class SinusoidalPositions(nn.Module):
    def __init__(self, width=32, maximum=128):
        super().__init__()
        self.width, self.maximum = width, maximum
        position = torch.arange(maximum).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        encoding = torch.zeros(maximum, width)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency[:width // 2])
        self.register_buffer('encoding', encoding)

    def forward(self, x):
        if x.shape[1] > self.maximum:
            raise ValueError('Sequence exceeds positional encoding capacity')
        return x + self.encoding[:x.shape[1]].to(dtype=x.dtype)


class TinySeq2Seq(nn.Module):
    """Teacher-forced logits; no pretrained weights or generation loop."""
    def __init__(self, width=32, heads=4, depth=1, vocabulary=32):
        super().__init__()
        self.width, self.heads, self.depth, self.vocabulary = width, heads, depth, vocabulary
        self.source_embedding = nn.Embedding(vocabulary, width)
        self.target_embedding = nn.Embedding(vocabulary, width)
        self.positions = SinusoidalPositions(width)
        self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(width, heads, width * 2,
            dropout=0.0, batch_first=True) for _ in range(depth)])
        self.decoder = nn.ModuleList([nn.TransformerDecoderLayer(width, heads, width * 2,
            dropout=0.0, batch_first=True) for _ in range(depth)])
        self.head = nn.Linear(width, vocabulary)

    def forward(self, source, target):
        memory = self.positions(self.source_embedding(source) * math.sqrt(self.width))
        for layer in self.encoder:
            memory = layer(memory)
        decoded = self.positions(self.target_embedding(target) * math.sqrt(self.width))
        mask = torch.ones(target.shape[1], target.shape[1], device=target.device, dtype=torch.bool).triu(1)
        for layer in self.decoder:
            decoded = layer(decoded, memory, tgt_mask=mask)
        return self.head(decoded)
