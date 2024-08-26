# Attention is all you need

Used the following resources to better understand this: 
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [pytorch-transformer github hkproj](https://github.com/hkproj/pytorch-transformer)
- [transformer github hyunwoongko](https://github.com/hyunwoongko/transformer)
- [datacamp build transformer with pytorch](https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch?dc_referrer=https%3A%2F%2Fwww.google.com%2F)

The model I coded is a combination of different parts i found in these resources so i understood the code well.

![transformer](https://cdn.analyticsvidhya.com/wp-content/uploads/2024/04/image12-removebg-preview-06-scaled.jpg)

## Features

- **Multi-Head Self-Attention**: Implements the multi-head self-attention mechanism that allows the model to focus on different parts of the input sequence simultaneously.
- **Positional Encoding**: Adds positional information to the input embeddings to capture the order of words in a sequence.
- **Layer Normalization**: Includes layer normalization after each sub-layer to stabilize and accelerate training.
- **Residual Connections**: Implements skip connections around each sub-layer to help with gradient flow.

## Files and Directories

- `blocks/`: Contains the core implementation of the Transformer model.
  - `decoder_layer.py`: Implements the decoder layer of the Transformer model.
  - `encoder_layer.py`: Implements the encoder layer of the Transformer model.
- `embedding/`: Handles the embedding operations.
  - `positional_encoding.py`: Implements positional encoding to provide order information to the embeddings.
- `layers/`: Contains individual components used within the Transformer model.
  - `feed_forward_network.py`: Implements the feed-forward neural network layer.
  - `layernorm.py`: Implements layer normalization.
  - `multi_head_attention.py`: Implements the multi-head attention mechanism.
  - `scaled_dot_product_attention.py`: Implements the scaled dot-product attention mechanism.
- `model/`: Contains the full Transformer model assembly.
  - `transformer.py`: Implements the complete Transformer model, assembling all the components.
