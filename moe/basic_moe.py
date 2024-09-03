import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """
    FFN layer, similar to the FFN used in Transformer Encoder layers.
    This class represents an individual expert in the Mixture of Experts model.
    """
    def __init__(self, model_dim, hidden_dim):
        super(Expert, self).__init__()
        self.input_layer = nn.Linear(model_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, model_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, model_dim)
        x = self.input_layer(x)  # x shape: (batch_size, seq_length, hidden_dim)
        x = self.relu(x) 
        x = self.output_layer(x)  # x shape: (batch_size, seq_length, model_dim)
        return x
    
class Router(nn.Module):
    """
    Router determines which experts should process each token.
    The router produces a probability distribution over the available experts.
    """
    def __init__(self, d_model, num_experts):
        super(Router, self).__init__()
        # Linear layer to produce routing scores for each expert
        self.linear_layer = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x shape: (batch_size, seq_length, d_model)
        # Compute routing weights and apply softmax to get probabilities
        return F.softmax(self.linear_layer(x), dim=-1)  # shape: (batch_size, seq_length, num_experts)

class MoE(nn.Module):
    """
    Mixture of Experts (MoE) layer, which dynamically routes tokens to a subset of experts.
    Each expert is an instance of a FeedForwardNetwork.
    """
    def __init__(self, model_dim, num_experts, hidden_dim, top_k=2):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([Expert(model_dim, hidden_dim) for _ in range(num_experts)])
        self.router = Router(model_dim, num_experts)
        self.top_k = top_k  # Number of top experts to select

    def forward(self, x):
        # x shape: (batch_size, seq_length, model_dim)
        # Determine routing weights using the router
        routing_weights = self.router(x)  # shape: (batch_size, seq_length, num_experts)
        # Select the top-k experts for each token
        topk_vals, topk_indices = torch.topk(routing_weights, self.top_k, dim=2)  # shape: (batch_size, seq_length, top_k)
        # Normalize the selected top-k weights
        topk_vals_normalized = topk_vals / topk_vals.sum(dim=2, keepdim=True)  # shape: (batch_size, seq_length, top_k)

        # Initialize the output tensor
        output = torch.zeros_like(x)  # shape: (batch_size, seq_length, model_dim)

        # Distribute the tokens to the selected experts
        for i, expert in enumerate(self.experts):
            # Create a mask for tokens routed to the current expert
            expert_mask = (topk_indices == i).float()  # shape: (batch_size, seq_length, top_k)
            if expert_mask.any():
                # Select tokens for the expert and apply the expert's transformation
                inputs_to_expert = x * expert_mask.unsqueeze(-1)  # shape: (batch_size, seq_length, model_dim)
                expert_output = expert(inputs_to_expert)  # shape: (batch_size, seq_length, model_dim)
                # Accumulate the expert outputs, weighted by the normalized top-k values
                output += expert_output * topk_vals_normalized.unsqueeze(-1)  # shape: (batch_size, seq_length, model_dim)

        return output  # shape: (batch_size, seq_length, model_dim)

class TransformerEncoderLayerWithMoE(nn.Module):
    """
    Transformer Encoder layer with a Mixture of Experts (MoE) component.
    """
    def __init__(self, model_dim, num_heads, num_experts, hidden_dim, dropout=0.1, top_k=2):
        super(TransformerEncoderLayerWithMoE, self).__init__()
        self.self_attention = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout)
        self.mixture_of_experts = MoE(model_dim, num_experts, hidden_dim, top_k=top_k)
        self.dropout = nn.Dropout(dropout) 
        self.norm1 = nn.LayerNorm(model_dim) 
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src shape: (batch_size, seq_length, model_dim)
        # Self-attention mechanism
        attn_output, _ = self.self_attention(src, src, src, attn_mask=src_mask,
                                             key_padding_mask=src_key_padding_mask)  # attn_output shape: (batch_size, seq_length, model_dim)
        # Add (residuals) & Norm for the self-attention output
        src = self.norm1(src + self.dropout(attn_output))  # shape: (batch_size, seq_length, model_dim)

        # Apply the Mixture of Experts layer
        moe_output = self.mixture_of_experts(src)  # shape: (batch_size, seq_length, model_dim)
        # Add (residuals) & Norm for the MoE output
        src = self.norm2(src + self.dropout(moe_output))  # shape: (batch_size, seq_length, model_dim)

        return src  # Final output shape: (batch_size, seq_length, model_dim)