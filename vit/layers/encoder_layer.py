import torch
import torch.nn as nn
from .layernorm import LayerNorm

import torch.nn as nn
from .layernorm import LayerNorm
from .multihead_attention import MultiheadAttention
from .mlp import MLP

class EncoderLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MultiheadAttention(dim, heads)
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, num_patches + 1, embedding_dim)
        x = x + self.dropout(self.attn(self.norm1(x))) # + because we want to add the residual connection
        x = x + self.dropout(self.mlp(self.norm2(x))) # + because we want to add the residual connection
        # Output shape: (batch_size, num_patches + 1, embedding_dim)
        return x