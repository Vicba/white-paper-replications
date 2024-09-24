import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_head == 0, "d_model must be divisible by number of heads"

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.dropout = nn.Dropout(0.1)

        self.W_query = nn.Linear(d_model, d_model)
        self.W_key = nn.Linear(d_model, d_model)
        self.W_value = nn.Linear(d_model, d_model)
        self.W_output = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1 do linear transformation
        batch_size = q.size(0)

        # Shape: (batch_size, seq_length, d_model)
        query = self.W_query(q)  # (batch_size, seq_length, d_model)
        key = self.W_key(k)      # (batch_size, seq_length, d_model)
        value = self.W_value(v)  # (batch_size, seq_length, d_model)

        # 2 reshape for multhihead attn
        query = query.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # (batch_size, n_head, seq_length, d_k)
        key = key.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)      # (batch_size, n_head, seq_length, d_k)
        value = value.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # (batch_size, n_head, seq_length, d_k)    

        # 3 attn scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)  # (batch_size, n_head, seq_length, seq_length)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, n_head, seq_length, seq_length)
        attn_weights = self.dropout(attn_weights)

        # 4 reshape and add values
        attn_output = torch.matmul(attn_weights, value)  # (batch_size, n_head, seq_length, d_k)

        # 5. combine heads heads and pass through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch_size, seq_length, n_head, d_k)
        attn_output = attn_output.view(batch_size, -1, self.d_model)  # (batch_size, seq_length, d_model)

        return self.W_output(attn_output) # (batch_size, seq_length, d_model)


if __name__ == "__main__":
    d_model = 768 
    n_head = 12

    multihead_attn = MultiHeadAttention(d_model, n_head)

    # Dummy input tensors (batch_size=2, seq_length=10)
    q = torch.rand(2, 10, d_model)
    k = torch.rand(2, 10, d_model)
    v = torch.rand(2, 10, d_model)

    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 
                         [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_length)

    output = multihead_attn(q, k, v, mask)
    print("Output shape:", output.shape)  # output (batch_size, seq_length, d_model)