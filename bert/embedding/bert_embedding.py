import math
import torch
import torch.nn as nn

# class PositionEmbedding(nn.Module):
#     def __init__(self, d_model, max_len):
#         super(PositionEmbedding, self).__init__()

#         pe = torch.zeros(max_len, d_model)
#         pos = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
#         )

#         pe[:, 0::2] = torch.sin(pos * div_term)
#         pe[:, 1::2] = torch.cos(pos * div_term)
#         pe = pe.unsqueeze(0) 
#         self.register_buffer("pe", pe)

#     def forward(self, x):
#         seq_len = x.size(1)  # Shape: (batch_size, seq_len, d_model)
#         return self.pe[:, :seq_len, :]  # Shape: (1, seq_len, d_model)

class AbsolutePositionEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(AbsolutePositionEmbedding, self).__init__()
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

    def forward(self, x): 
        batch_size, max_seq_len = x.size()
        positions = torch.arange(0, max_seq_len).unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, max_seq_len)
        return self.pos_emb(positions) # shape (batch_size, seq_len, d_model)


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len=64, dropout=0.1):
        super(BERTEmbedding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        self.seg_embedding = nn.Embedding(3, d_model)  # Shape: (3, d_model), 3 bc: 1 = sent1, 2 = sent2, 3 = pad
        self.tok_embedding = nn.Embedding(vocab_size, d_model)  # Shape: (vocab_size, d_model)
        self.pos_embedding = AbsolutePositionEmbedding(d_model, max_seq_len=seq_len)  # Shape: (1, seq_len, d_model)

    def forward(self, sequence, segment_labels):
        x = self.tok_embedding(sequence) + self.pos_embedding(sequence) + self.seg_embedding(segment_labels)  # Shape: (batch_size, seq_len, d_model)
        return self.dropout(x)  # Shape: (batch_size, seq_len, d_model)
    

if __name__ == "__main__":
    d_model = 768
    max_seq_len = 64
    seq_len = 64
    vocab_size = 30000
    batch_size = 32

    # Create random input data
    sequence = torch.randint(0, vocab_size, (batch_size, seq_len))  # Random token indices
    segment_labels = torch.randint(0, 3, (batch_size, seq_len))  # Random segment labels (0, 1, or 2)
    print("sequence shape: ", sequence.shape)
    print("segment labels shape: ", segment_labels.shape)

    # test absolute pos embedding
    emb_model = AbsolutePositionEmbedding(d_model, max_seq_len)
    embeddings = emb_model(sequence)
    print("embeddings shape: ", embeddings.shape)

    # test bert embedding with absolute pos emb
    bert_emb = BERTEmbedding(vocab_size, d_model, seq_len)
    embeddings_bert = bert_emb(sequence, segment_labels)
    print("emb bert: ", embeddings_bert.shape)

    assert embeddings_bert.shape == (batch_size, seq_len, d_model), f"Expected embeddings_bert shape {(batch_size, seq_len, d_model)}, got {embeddings_bert.shape}"    
