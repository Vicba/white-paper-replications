import torch
import torch.nn as nn
from utils.gelu import GELU

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = GELU()

    def forward(self, x):
        return self.fc2(self.dropout(self.gelu(self.fc1(x))))