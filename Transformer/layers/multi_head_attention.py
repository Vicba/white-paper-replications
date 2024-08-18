import torch
import torch.nn as nn
from transformer.layers.scaled_dot_product_attention import ScaleDotProductAttention

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

        # Linear layers for transforming inputs into query (Q), key (K), and value (V) vectors
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation

    def split_heads(self, x):
        """
        Split the input tensor into multiple heads for parallel attention computation.

        :param x: Input tensor of shape [batch_size, seq_length, d_model]
        :return: Tensor of shape [batch_size, n_heads, seq_length, d_k]
        """
        # Reshape the tensor to separate the attention heads and adjust dimensions accordingly
        # New shape: [batch_size, seq_length, n_heads, d_k]
        # Then, transpose to bring n_heads to the second dimension: [batch_size, n_heads, seq_length, d_k]
        # Original shape: torch.Size([1, 100, 512])
        # Reshaped shape: torch.Size([1, 100, 8, 64])
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        Combine the multiple attention heads back into a single tensor.

        :param x: Tensor of shape [batch_size, n_heads, seq_length, d_k]
        :return: Tensor of shape [batch_size, seq_length, d_model]
        """
        batch_size, n_heads, seq_length, d_k = x.size()
        # Transpose to bring the seq_length back to its original position
        # Contiguous ensures the tensor's memory layout is suitable for reshaping
        # Then, flatten the n_heads and d_k dimensions back to the original d_model dimension
        # Final shape: [batch_size, seq_length, d_model]
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Forward pass for the multi-head attention mechanism.

        :param Q: Query tensor of shape [batch_size, seq_length, d_model]
        :param K: Key tensor of shape [batch_size, seq_length, d_model]
        :param V: Value tensor of shape [batch_size, seq_length, d_model]
        :param mask: Optional mask to prevent attention to certain positions
        :return: Tuple (output, attn_weights) where output is the attention result 
                 and attn_weights are the attention weights
        """
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q)) # results in: (batch_size, n_heads, seq_length, d_k)
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perform scaled dot-product attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply final linear output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output, attn_weights

