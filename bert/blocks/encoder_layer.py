import torch
import torch.nn as nn
from layers.multiheadattention import MHA
from layers.feed_forward import PositionwiseFeedForward
from layers.layernorm import LayerNorm

class EncoderLayer(nn.Module):
    def __init__(self, d_model=768, n_head=12, d_ff=768 * 4, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.layernorm = nn.LayerNorm(d_model)
        self.self_multihead = MHA(n_head, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings, mask):
        # embeddings: (batch_size, max_len, d_model)
        # encoder mask: (batch_size, 1, 1, max_len)
        # result: (batch_size, max_len, d_model)
        out_mha = self.dropout(self.self_multihead(embeddings, mask))
        # add and norm layer
        out_mha = self.layernorm(out_mha + embeddings)

        # ffn layer in encoder block
        out_ffn = self.dropout(self.feed_forward(out_mha))
        encoded = self.layernorm(out_ffn + out_mha)
        return encoded