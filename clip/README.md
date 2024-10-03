# CLIP

## TODO
- test dataset.py
- test train.py

CLIP (Contrastive Language-Image Pretraining) is a neural network trained on a dataset of 400 million pairs of images and text, where only 20,000 pairs are correctly matched. The model learns to associate images and their corresponding text descriptions by maximizing the similarity between their vector representations.

**Model Architecture**

CLIP utilizes two distinct encoders:
- **Text Encoder**: BERT
- **Image Encoder**: Typically a Vision Transformer (ViT), but can also be ResNet

Both encoders output single vector embeddings for each input, resulting in multi-dimensional vectors that can be compared within the same vector space.

![](https://miro.medium.com/v2/resize:fit:3662/1*tg7akErlMSyCLQxrMtQIYw.png)

Source:
- [Understanding contrastive learning](https://towardsdatascience.com/understanding-contrastive-learning-d5b19fd96607)
- [Medium explenation + implementation](https://towardsdatascience.com/simple-implementation-of-openai-clip-model-a-tutorial-ace6ff01d9f2)
- [Medium explenation + implementation](https://towardsdatascience.com/clip-model-and-the-importance-of-multimodal-embeddings-1c8f6b13bf72)
- [Huggingface](https://huggingface.co/docs/transformers/model_doc/clip)
- [HF github transformers library](https://github.com/huggingface/transformers/tree/main/src/transformers/models/clip)
- [OPENAI CLIP (github)](https://github.com/openai/CLIP/tree/main/clip)


## Notebook

I also found a cool notebook that showcases some explainability (tried to add some comments to).

