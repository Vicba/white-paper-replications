import torch
import torch.nn as nn
from layers.scaled_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.n_heads = n_heads # Number of attention heads
        self.d_k = d_model // n_heads # Dimension of each head's key, query, and value

        self.scaled_dot_product_attention = ScaleDotProductAttention()

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation

    def split_heads(self, x):
        """
        split tensor by number of head

        :param tensor: [batch_size, seq_length, d_model]
        :return: [batch_size, n_head, seq_length, d_k]
        """
        # Reshape the input to have n_heads for multi-head attention
        # Original shape: torch.Size([1, 100, 512])
        # Reshaped shape: torch.Size([1, 100, 8, 64])
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        combine multiple heads

        :param tensor: [batch_size, n_head, seq_length, d_k]
        :return: [batch_size, length, d_model]
        """
        # Combine the multiple heads back to original shape
        batch_size, n_heads, seq_length, d_k = x.size()
        # contiguous() ensures the tensor's memory layout is suitable for reshaping, 
        # and view(batch_size, seq_length, d_model) flattens the n_heads and d_k dimensions back 
        # into the original d_model dimension, resulting in a tensor of shape (batch_size, seq_length, d_model).
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q)) # results in: (batch_size, n_heads, seq_length, d_k)
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output