# Visual Transformer (ViT)

This is a replication of the ViT paper. A visual transformer (ViT) is a type of neural network that uses self-attention to process images for classification tasks.

![vit](https://velog.velcdn.com/images/uvictoli/post/9a900600-956e-4254-bd95-6239316b317a/image.png)

## Run code

```bash
cd vit
source venv/bin/activate
pip install -r requirements.txt
```

```bash
# test code
python test.py
# train and predict
python train_predict.py
```

## Files and Directories

- `model/`: Contains the main ViT model implementation.
  - `vit.py`: Implements the Visual Transformer (ViT) model.
- `layers/`: Contains individual components of the ViT architecture.
  - `encoder_layer.py`: Implements the Transformer encoder layer, which includes self-attention and feed-forward networks with residual connections and layer normalization.
  - `layernorm.py`: Implements Layer Normalization, which normalizes the inputs across the features, helping to stabilize and accelerate training.
  - `mlp.py`: Implements the Multi-Layer Perceptron, a feed-forward network used after the attention mechanism in each encoder layer.
  - `multihead_attention.py`: Implements the Multi-Head Attention mechanism, allowing the model to jointly attend to information from different representation subspaces.
  - `patch_embedding.py`: Implements the Patch Embedding layer, which splits the input image into fixed-size patches and linearly embeds each patch.
- `test.py`: Contains a simple test script to verify the ViT model's output shape.
- `train_predict.py`: Implements training and prediction functions for the ViT model using the CIFAR-10 dataset.
- `requirements.txt`: Lists the required Python packages for the project.