import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-05):
        super(LayerNorm, self).__init__()
        
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
        # Small epsilon value for numerical stability to avoid division by zero during normalization
        self.eps = eps

    def forward(self, x):
        # Compute the mean of the input tensor x along the last dimension (usually the feature dimension)
        # keepdim=True keeps the dimensions intact for broadcasting
        mean = x.mean(dim=-1, keepdim=True)
        
        # Compute the variance of the input tensor x along the last dimension
        # unbiased=False ensures the calculation uses the biased estimator, which divides by N instead of N-1
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        # Normalize the input: subtract the mean and divide by the square root of the variance plus epsilon
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)

        # Apply the learnable scale (gamma) and shift (beta) to the normalized input
        out = self.gamma * normalized_x + self.beta

        return out
