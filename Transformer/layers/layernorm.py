import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-05):
        super(LayerNorm, self).__init__()
        # Scale and shift parameters, initialized to ones and zeros respectively
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        # Small epsilon value for numerical stability
        self.eps = eps

    def forward(self, x):
        # Compute the mean and variance across the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        # Normalize the input
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        out = self.gamma * normalized_x + self.beta
        return out
