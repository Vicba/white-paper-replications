import torch
import torch.nn as nn
from layers.layernorm import LayerNorm
from layers.feed_forward import PositionwiseFeedForward
from layers.multiheadattention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.mh_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # attn with residual connectinos and layer norm
        attn_output = self.mh_attn(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout1(attn_output))

        # ffn with residual connections and layer norm
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + self.dropout2(ffn_output))

        return x
    
if __name__ == "__main__":
    d_model = 768
    n_head = 12
    d_ff = d_model * 4

    encoder_layer = EncoderLayer(d_model, n_head, d_ff)

    # Dummy input tensor (batch_size=2, seq_length=10)
    x = torch.rand(2, 10, d_model)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 
                         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_length)

    output = encoder_layer(x, mask)
    print("Encoder Layer Output shape:", output.shape)  # Should output (batch_size, seq_length, d_model)
