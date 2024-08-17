import torch
import torch.nn as nn
from transformer.layers.multi_head_attention import MultiHeadAttention
from transformer.layers.feed_forward_network import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 1. compute self attention
        attn_output = self.attn(Q=x, K=x, V=x, mask=tgt_mask)
        # 2. add & norm
        x = self.norm1(x + self.dropout(attn_output))
        # 3. compute encoder - decoder attention
        attn_output = self.cross_attn(Q=x, K=enc_output, V=enc_output, mask=src_mask)
        # 4. add & norm
        x = self.norm2(x + self.dropout(attn_output))
        # 5. positionwise feed forward network
        ff_output = self.feed_forward(x)
        # 6. add and norm
        x = self.norm3(x + self.dropout(ff_output))
        return x