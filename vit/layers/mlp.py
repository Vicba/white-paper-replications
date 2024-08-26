import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: (batch_size, num_patches, embedding_dim)
        # Output shape: (batch_size, num_patches, embedding_dim)
        return self.net(x)