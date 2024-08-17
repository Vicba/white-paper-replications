import torch
import torch.nn as nn
import pytest
from Transformer.layers.layernorm import LayerNorm

def test_layernorm(batch_size, seq_len, d_model):
    # Set a seed for reproducibility
    torch.manual_seed(42)

    # Generate random input tensor
    x = torch.randn(batch_size, seq_len, d_model)

    # Create instances of both the custom LayerNorm and the built-in LayerNorm
    custom_ln = LayerNorm(d_model)
    pytorch_ln = nn.LayerNorm(d_model)

    # Manually copy the parameters to ensure they start with the same values
    with torch.inference_mode():
        pytorch_ln.weight.copy_(custom_ln.gamma)
        pytorch_ln.bias.copy_(custom_ln.beta)

    # Get the outputs
    custom_output = custom_ln(x)
    pytorch_output = pytorch_ln(x)

    # Assert that the outputs are close
    assert torch.allclose(custom_output, pytorch_output, atol=1e-6), \
        f"Custom LayerNorm output differs from built-in LayerNorm output!"

if __name__ == "__main__":
    pytest.main()
