import torch
import torch.nn as nn
from layers.encoder_layer import EncoderLayer
from layers.patch_embedding import PatchEmbedding

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, n_layers, heads, mlp_dim, dropout=0.1):
        super().__init__()
        # Convert image into patches and embed them
        # Output shape: (batch_size, num_patches, dim)
        self.patch_embedding = PatchEmbedding(image_size, patch_size, dim)
        
        # Learnable class token
        # Shape: (1, 1, dim)
        self.class_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Learnable 1D position embeddings
        # Shape: (1, num_patches + 1, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_embedding.num_patches + 1, dim)) # +1 for the class token: (1, 197, 768)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Stack of Transformer encoder layers
        # Each layer processes: (batch_size, num_patches + 1, dim)
        self.encoder = nn.Sequential(*[EncoderLayer(dim, heads, mlp_dim, dropout) for _ in range(n_layers)])
        
        # Final classification head
        # Input: (batch_size, dim), Output: (batch_size, num_classes)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # img shape: (batch_size, channels, height, width)
        x = self.patch_embedding(img)
        b, n, _ = x.shape # img with shape (batch_size, num_patches, dim)

        # Expand class token to batch size
        # creates copies of this single token for each image in the current batch.
        # class_tokens shape: (batch_size, 1, dim)
        # first -1 means "keep the second dimension as is" (which is 1)
        # second -1 means "keep the third dimension as is" (which is dim)
        class_tokens = self.class_token.expand(b, -1, -1)
        
        # Concatenate class token with patch embeddings
        # x shape: (batch_size, num_patches + 1, dim)
        x = torch.cat((class_tokens, x), dim=1)
        
        # Add positional embeddings
        x += self.pos_embedding
        
        # Apply dropout
        x = self.dropout(x)

        # Pass through encoder layers
        # x shape remains: (batch_size, num_patches + 1, dim)
        x = self.encoder(x)
        
        # Extract class token representation
        # x shape: (batch_size, dim)
        x = x[:, 0]

        # Final classification
        # Output shape: (batch_size, num_classes)
        return self.mlp_head(x)