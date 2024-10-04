# CLIP

CLIP (Contrastive Language-Image Pretraining) is a neural network trained on a dataset of 400 million pairs of images and text, where only 20,000 pairs are correctly matched. The model learns to associate images and their corresponding text descriptions by maximizing the similarity between their vector representations.

**Model Architecture**

CLIP utilizes two distinct encoders:
- **Text Encoder**: BERT
- **Image Encoder**: Typically a Vision Transformer (ViT), but can also be ResNet

Both encoders output single vector embeddings for each input, resulting in multi-dimensional vectors that can be compared within the same vector space.

![](https://miro.medium.com/v2/resize:fit:3662/1*tg7akErlMSyCLQxrMtQIYw.png)

Sources to learn CLIP:
- [Understanding contrastive learning](https://towardsdatascience.com/understanding-contrastive-learning-d5b19fd96607)
- [explenation on YT](https://www.youtube.com/watch?v=jXD6O93Ptks&t=231s)
- [YT explenation (james briggs](https://www.youtube.com/watch?v=98POYg2HZqQ)
- [Huggingface](https://huggingface.co/docs/transformers/model_doc/clip)
- [HF github transformers library](https://github.com/huggingface/transformers/tree/main/src/transformers/models/clip)
- [OPENAI CLIP (github)](https://github.com/openai/CLIP/tree/main/clip)


## Notebook

I also found a cool notebook that showcases some explainability (tried to add some comments).

