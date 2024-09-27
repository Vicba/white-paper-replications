import math
import torch
import torch.nn as nn

class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super(PositionEmbedding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model) for broadcasting
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)  # Get sequence length from input x
        return self.pe[:, :seq_len, :]  # Return corresponding positional encodings

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len=64, dropout=0.1):
        super(BERTEmbedding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        self.segm_emb = nn.Embedding(3, d_model)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionEmbedding(d_model, max_len=seq_len)

    def forward(self, sequence, segment_labels):
        x = self.tok_emb(sequence) + self.pos_emb(sequence) + self.segm_emb(segment_labels)
        return self.dropout(x)