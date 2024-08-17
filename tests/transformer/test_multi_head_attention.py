import pytest
import torch
import torch.nn as nn
from Transformer.layers.multi_head_attention import MultiHeadAttention


# Function to create random input tensors
def create_inputs(batch_size, seq_len, embed_dim):
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)
    return query, key, value


def test_multihead_attention_no_mask(batch_size, seq_len, embed_dim, num_heads):
    torch.manual_seed(42)  # Set seed for reproducibility
    
    # Create input tensors
    query, key, value = create_inputs(batch_size, seq_len, embed_dim)
    
    # Initialize your custom MultiHeadAttention class and PyTorch's built-in MultiheadAttention
    custom_mha = MultiHeadAttention(embed_dim, num_heads)
    pytorch_mha = nn.MultiheadAttention(embed_dim, num_heads)
    
    # Get the output from both models without mask
    custom_output, custom_attn_weights = custom_mha(query, key, value)
    pytorch_output, pytorch_attn_weights = pytorch_mha(query, key, value)

    # Assert that the outputs are nearly equal
    assert torch.allclose(custom_output, pytorch_output, atol=1e-6), "Outputs differ (no mask)"
    assert torch.allclose(custom_attn_weights, pytorch_attn_weights, atol=1e-6), "Attention weights differ (no mask)"



def test_multihead_attention_with_mask(batch_size, seq_len, embed_dim, num_heads):
    torch.manual_seed(42)  # Set seed for reproducibility
    
    # Create input tensors
    query, key, value = create_inputs(batch_size, seq_len, embed_dim)
    mask = torch.randint(0, 2, (batch_size, seq_len)).bool()

    # Initialize your custom MultiHeadAttention class and PyTorch's built-in MultiheadAttention
    custom_mha = MultiHeadAttention(embed_dim, num_heads)
    pytorch_mha = nn.MultiheadAttention(embed_dim, num_heads)
    
    # Get the output from both models with mask
    custom_output, custom_attn_weights = custom_mha(query, key, value, mask)
    pytorch_output, pytorch_attn_weights = pytorch_mha(query, key, value, attn_mask=mask)

    # Assert that the outputs are nearly equal
    assert torch.allclose(custom_output, pytorch_output, atol=1e-6), "Outputs differ (with mask)"
    assert torch.allclose(custom_attn_weights, pytorch_attn_weights, atol=1e-6), "Attention weights differ (with mask)"



if __name__ == "__main__":
    pytest.main()
