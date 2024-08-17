import torch
import torch.nn as nn
from transformer.blocks.encoder_layer import EncoderLayer
from transformer.blocks.decoder_layer import DecoderLayer
from transformer.embedding.positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()

        # Embedding layers for source and target sequences
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)  # Embedding for source tokens
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)  # Embedding for target tokens

        # Positional encoding to provide sequence information
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Stack of encoder layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        # Stack of decoder layers
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])

        # Final linear layer to transform the decoder output to target vocabulary size
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        # Create a mask for the source sequence to ignore padding tokens
        # src != 0 creates a boolean tensor where padding tokens (zeros) are marked as False and non-padding tokens are marked as True
        # unsqueeze(1) adds an extra dimension to make room for attention heads
        # unsqueeze(2) adds another dimension to match the expected shape for attention mechanisms
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, src_seq_len)

        # Create a mask for the target sequence to ignore padding tokens
        # tgt != 0 creates a boolean tensor similar to the source mask
        # unsqueeze(1) and unsqueeze(3) prepare the mask for broadcasting in the decoder
        # tgt_mask will be used to prevent attending to padding tokens in the target sequence
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)  # Shape: (batch_size, 1, tgt_seq_len, 1)

        # Create a no-peek mask to ensure that each position can only attend to earlier positions in the target sequence
        # This is crucial for autoregressive decoding where future tokens should not influence the prediction of the current token
        seq_length = tgt.size(1)  # Length of the target sequence
        # Generate an upper triangular matrix with ones on and above the diagonal and zeros below it
        # This matrix will be used to mask future positions (set them to -inf or 0 depending on implementation)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()  # Shape: (1, tgt_seq_len, tgt_seq_len)

        # Combine the padding mask and the no-peek mask
        # tgt_mask ensures that padding tokens are not attended to
        # nopeak_mask ensures that future positions are not attended to during decoding
        tgt_mask = tgt_mask & nopeak_mask  # Shape: (batch_size, 1, tgt_seq_len, tgt_seq_len)

        return src_mask, tgt_mask

    def forward(self, src, tgt):
        # Generate masks for source and target sequences
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # Embed and apply positional encoding to source and target sequences
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        # Pass the source embeddings through the encoder layers
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # Pass the target embeddings through the decoder layers
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        # Apply final linear layer to obtain predictions for target vocabulary
        output = self.fc(dec_output)
        return output
