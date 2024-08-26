import torch.nn as nn
import torch.nn.functional as F
import torch

class MultiheadAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5  # same as sqrt(d_k)

        # Linear projection for query, key, and value (combined)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        # Linear projection for output
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        # x shape: (batch_size, num_patches, embedding_dim)
        b, n, c = x.shape

        # Project input to query, key, and value
        # qkv shape: (batch_size, num_patches, 3 * embedding_dim)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # Reshape and transpose for multi-head attention
        # q, k, v shape: (batch_size, num_heads, num_patches, head_dim)
        q, k, v = map(lambda t: t.reshape(b, n, self.heads, c // self.heads).transpose(1, 2), qkv)

        # Compute scaled dot-product attention
        # dots shape: (batch_size, num_heads, num_patches, num_patches)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # Apply softmax to get attention weights
        attn = F.softmax(dots, dim=-1)

        # Apply attention weights to values
        # out shape: (batch_size, num_heads, num_patches, head_dim)
        out = torch.matmul(attn, v)
        # Reshape and transpose back
        # out shape: (batch_size, num_patches, embedding_dim)
        out = out.transpose(1, 2).reshape(b, n, c)
        # Final linear projection
        return self.to_out(out)