import torch
import torch.nn as nn
import math

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None):
        batch_size, n_heads, seq_length, d_k = K.size()
        
        # Calculate attention scores by performing a matrix multiplication between Q and the transpose of K
        # Transpose the last two dimensions of K so that the shapes align for matrix multiplication
        # Q has shape (batch_size, n_heads, seq_length, d_k)
        # K.transpose(-2, -1) changes K to shape (batch_size, n_heads, d_k, seq_length)
        # changing shape from (batch_size, n_heads, seq_len, d_k) to (batch_size, n_heads, d_k, seq_len)
        # Resulting attn_scores will have shape (batch_size, n_heads, seq_length, seq_length)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply the mask (if provided) to the attention scores
        # This is useful to prevent the model from attending to certain positions, such as padding tokens in sequences
        # The mask will set the attention scores of the masked positions to a very large negative value (-1e9)        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax is applied to obtain attention probabilities
        # Softmax is applied along the last dimension (seq_length) to get a probability distribution over the sequence
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Calculate the final output by performing a weighted sum of the values (V) using the attention probabilities
        # Multiply attn_probs (which has shape [batch_size, n_heads, seq_length, seq_length])
        # by V (which has shape [batch_size, n_heads, seq_length, d_k]) to get the output
        output = torch.matmul(attn_probs, V)
        return output, attn_probs