import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
  def __init__(self, n_head, d_model, dropout=0.1):
    super(MHA, self).__init__()
    assert d_model % n_head == 0

    self.d_k = d_model // n_head
    self.n_head = n_head
    self.d_model = d_model
    self.dropout = nn.Dropout(p=dropout)

    self.W_q = nn.Linear(d_model, d_model)
    self.W_k = nn.Linear(d_model, d_model)
    self.W_v = nn.Linear(d_model, d_model)
    self.W_o = nn.Linear(d_model, d_model)

  def forward(self, x, mask=None):
    batch_size, max_len, d_model = x.size()

    # Linear projection and reshape to (batch_size, max_len, n_head, d_k)
    query = self.W_q(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2) # (batch_size, n_head, max_len, d_k)
    key = self.W_k(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2) # (batch_size, n_head, max_len, d_k)
    value = self.W_v(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2) # (batch_size, n_head, max_len, d_k)

    # Scaled dot-product attention: (batch_size, n_head, max_len, max_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

    if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e9)

    weights = F.softmax(scores, dim=-1) # (batch_size, n_head, max_len, max_len)
    weights = self.dropout(weights)

    context = torch.matmul(weights, value) # (batch_size, n_head, max_len, d_k)
    context = context.transpose(1, 2).contiguous().view(batch_size, -1, d_model) # Reshape to (batch_size, max_len, d_model)

    return self.W_o(context)