import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x shape: (batch_size, num_patches, embedding_dim)
        # Compute mean and standard deviation along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        # Normalize the input
        normalized = (x - mean) / (std + 1e-6)
        
        # Apply learnable parameters gamma and beta
        # Output shape: (batch_size, num_patches, embedding_dim)
        return normalized * self.gamma + self.beta