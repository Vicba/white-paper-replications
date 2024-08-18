import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        # Create a tensor to hold the positional encodings, initialized to zeros
        # Shape: (max_seq_length, d_model), where d_model is the embedding size
        self.pe = torch.zeros(max_seq_length, d_model)

        # Create a tensor of positions from 0 to max_seq_length-1 (for each position in the sequence)
        # Unsqueeze to add a second dimension, making the tensor shape (max_seq_length, 1)
        pos = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # Create a tensor for the even indices of the d_model (e.g., if d_model=50, _2i=[0, 2, 4, ..., 48])
        # The step=2 ensures we get only even indices
        _2i = torch.arange(0, d_model, step=2).float()

        # Calculate the positional encodings using sine and cosine functions
        self.pe[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        # Add the positional encodings to the input embeddings
        # Use the first x.size(1) elements to match the sequence length of the input x
        return x + self.pe[:, :x.size(1)]
