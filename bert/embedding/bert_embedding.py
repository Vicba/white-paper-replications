import math
import torch
import torch.nn as nn
from tokenizers import BertWord


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len, dropout=0.1):
        super(BERTEmbedding, self).__init__()
        
        self.vocab_size = vocab_size
        self.token_embeddings = TokenEmbedding(vocab_size, d_model)
        self.pos_embeddings = PositionEmbedding(max_seq_len, d_model)
        self.segment_embeddings = SegmentEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, segment_ids):
        token_emb = self.token_embeddings(input_ids)
        pos_emb = self.pos_embeddings(input_ids)
        segment_emb = self.segment_embeddings(segment_ids)

        input_embeddings = token_emb + pos_emb + segment_emb
        return self.dropout(input_embeddings)

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

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x): # shape: (batch_size, seq_len)
        return self.token_embeddings(x)

class SegmentEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SegmentEmbedding, self).__init__()
        self.segment_embeddings = nn.Embedding(3, d_model) # 3 bc: sent1, sent2 and padding

    def forward(self, segments): # shape (batch_size, seq_len)
        return self.segment_embeddings(segments) # shape: (batch_size, seq_len, embedding_dim)
    

if __name__ == "__main__":
    embed_layer = BERTEmbedding(vocab_size=len(tokenizer.vocab), d_model=768, seq_len=MAX_LEN)
    embed_result = embed_layer(sample_data["bert_input"], sample_data["segment_label"])
    print(embed_result.size())