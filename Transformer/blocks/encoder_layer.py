import torch
import torch.nn as nn
from transformer.layers.multi_head_attention import MultiHeadAttention
from transformer.layers.feed_forward_network import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout): # d_ff: The dimensionality of the inner layer in the position-wise feed-forward network.
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # compute self attention
        attn_output = self.attn(Q=x, K=x, V=x, mask=mask)
        # add & norm
        x = self.norm1(x + self.dropout(attn_output))
        # FFN
        ff_output = self.feed_forward(x)
        # add & norm
        x = self.norm2(x + self.dropout(ff_output))
        return x