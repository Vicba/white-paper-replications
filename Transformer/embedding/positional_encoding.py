import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        # pe: A tensor filled with zeros, which will be populated with positional encodings.
        self.pe = torch.zeros(max_seq_length, d_model)

        # position: A tensor containing the position indices for each position in the sequence.
        # # 1D => 2D unsqueeze to represent word's position
        pos = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.pe[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model))) # 0::2: selects all even indices
        self.pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
      # uses the first x.size(1) elements of pe to ensure that the positional encodings match the actual sequence length of x.
      return x + self.pe[:, :x.size(1)] # add positional encodings to input x