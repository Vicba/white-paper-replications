import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2 # 196 => 14x14
        # Convolutional layer to create patch embeddings
        self.proj = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        # Apply convolution to create patch embeddings
        # x shape after conv: (batch_size, dim, height // patch_size, width // patch_size)
        x = self.proj(x)
        # Flatten and transpose to get final patch embeddings
        # Output shape: (batch_size, num_patches, dim)
        return x.flatten(2).transpose(1, 2)