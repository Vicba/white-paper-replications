import torch
import torch.nn as nn
from blocks.encoder_layer import EncoderLayer
from embedding.bert_embedding import BERTEmbedding

class BERT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_head, max_seq_len, dropout=0.1):
        super(BERT, self).__init__()

        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_ff = d_model * 4

        self.embedding = BERTEmbedding(vocab_size, d_model, max_seq_len, dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, self.d_ff, dropout) for _ in range(n_layer)])

    def forward(self, input_ids, segment_ids, mask):
        x = self.embedding(input_ids, segment_ids)

        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        return x
    
if __name__ == "__main__":
    vocab_size = 30522
    d_model = 768
    n_head = 12
    n_layers = 12
    max_seq_len = 512

    model = BERT(vocab_size, d_model, n_head, n_layers, max_seq_len)

    # Dummy input tensors
    input_ids = torch.tensor([[101, 2054, 2003, 1996, 5270, 102, 0, 0, 0, 0],
                               [101, 2009, 2003, 1037, 2154, 102, 0, 0, 0, 0]])
    segment_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 
                         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_length)

    output = model(input_ids, segment_ids, mask)
    print("BERT output", output)
    print("BERT output shape:", output.shape)  # Should output (batch_size, seq_length, d_model)

