import torch
import torch.nn as nn
import math

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None):
        batch_size, n_heads, seq_length, d_k = K.size()
        
        # Calculate attention scores
        # transposes the last two dimensions of the key matrix K, 
        # changing shape from (batch_size, num_heads, seq_len, d_k) to (batch_size, num_heads, d_k, seq_len)
        # attn_scores with the shape (batch_size, num_heads, seq_len, seq_len).
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax is applied to obtain attention probabilities
        # Apply softmax along the last dimension (seq_len)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output