import torch
import torch.nn as nn
from utils.gelu import GELU
from layers.layernorm import LayerNorm

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, dropout=0.1):
        super(BERTEmbedding, self).__init__()
        
        self.token_embeddings = TokenEmbedding(vocab_size, d_model)
        self.pos_embeddings = PositionEmbedding(max_seq_len, d_model)
        self.segment_embeddings = SegmentEmbedding(d_model)
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, segment_ids):
        token_emb = self.token_embeddings(input_ids)
        pos_emb = self.pos_embeddings(input_ids)
        segment_emb = self.segment_embeddings(segment_ids)

        embeddings = token_emb + pos_emb + segment_emb

        return self.dropout(self.layer_norm(embeddings))

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x): # shape: (batch_size, seq_len)
        return self.token_embeddings(x)

class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super(PositionEmbedding, self).__init__()
        
        self.pos_embeddings = nn.Embedding(max_seq_len, d_model)

    def forward(self, x): # shape: (batch_size, max_seq_len)
        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand_as(x) # (seq_len,) -> (batch_size, seq_len)
        return self.pos_embeddings(position_ids)


class SegmentEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SegmentEmbedding, self).__init__()
        self.segment_embeddings = nn.Embedding(2, d_model)

    def forward(self, segments): # shape (batch_size, seq_len)
        return self.segment_embeddings(segments) # shape: (batch_size, seq_len, embedding_dim)
    

if __name__ == "__main__":
    vocab_size = 30522
    d_model = 768
    max_seq_len = 512

    bert_embedding_layer = BERTEmbedding(vocab_size, d_model, max_seq_len)

    # Dummy input_ids tensor (batch_size=2, seq_len=10)
    input_ids = torch.tensor([[101, 2054, 2003, 1996, 5270, 102, 0, 0, 0, 0],
                              [101, 2009, 2003, 1037, 2154, 102, 0, 0, 0, 0]])

    # Dummy segment_ids tensor (batch_size=2, seq_len=10), 0 for the first segment, 1 for the second
    segment_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])

    embeddings = bert_embedding_layer(input_ids, segment_ids)

    print("Embeddings:", embeddings)
    print("Embeddings shape:", embeddings.shape)  # output (batch_size, seq_len, embedding_dim)