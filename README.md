# White Paper Replicas

Welcome to the **White Paper Replications** repository! This project is dedicated to replicating state-of-the-art AI research from influential white papers, primarily using PyTorch and Python.

`Disclaimer:` The code is written by me, so it's not perfect. It's a combination of my own insights and parts I found in different resources.

Planning on adding these in the future
SAM, RoPE, FlashAttention, Whisper, UNET, DDDM, DDIM, BLOOM

## Overview

In this repository, you'll find implementations of various AI models and algorithms as described in prominent research papers. The goal is to better understand them by coding and trying to make them as intuitive as possible.

## Available Replicas

Here's a list of the white papers and their corresponding implementations available in this repository:

- **["Attention is All You Need" paper](https://arxiv.org/abs/1706.03762)**: Introduces the Transformer model, which uses self-attention to focus on important words in a sentence, making it faster and better at understanding long sentences compared to older models.
- **["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929)**: Presents the Visual Transformer (ViT), a neural network that applies self-attention to image classification tasks.
- **Mixture of Experts (various papers)**: Architecture that uses multiple specialized models (experts) and a gating mechanism to improve performance and adaptability by activating only the most relevant experts for a given task.
- **["(Word2Vec) Efficient Estimation of Word Representations in Vector Space"](https://arxiv.org/pdf/1301.3781)**: Technique that transforms words into high-dimensional vectors, capturing their meaning and relationships based on the context in which they appear.
- **["GloVe: Global Vectors for Word Representation"](https://nlp.stanford.edu/pubs/glove.pdf)**: Technique used in NLP that captures global context information for creating word embeddings.
- **["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/pdf/1810.04805)**: BERT is a transformer-based model that understands word context in sentences by analyzing text bidirectionally.
- **["CLIP: Learning Transferable Visual Models From Natural Language Supervision"](https://arxiv.org/pdf/2103.00020)**: Multimodal neural network that learns to connect images and text by training an image encoder and text encoder to maximize the similarity between correct image-text pairs while minimizing it for incorrect pairs.
- **["U-net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/pdf/1505.04597)**: The U-Net architecture consists of an encoder that downsamples the input image to capture features and a decoder that upsamples it to produce a segmentation map, forming a "U" shape. Skip connections between corresponding layers in the encoder and decoder help preserve spatial information, increasing the model's performance in segmentation tasks.

Each one contains:
- **`README.md`**: Detailed instructions on how to use the code for that specific paper.
<!-- - **`main.py`**: The main script to run the implementation. -->
- **`requirements.txt`**: Python dependencies required for that implementation.
<!-- - **`data/`**: Dataset and other necessary files (if applicable). -->

## Table of Contents

- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

To get started with the code in this repository, follow these steps:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Vicba/white-paper-replications.git
   cd white-paper-replications
   ```
   
## Requirements

To run the code, you'll need:

- Python

## Usage

To view/use the code of a specific paper, navigate to the corresponding directory and follow the instructions in the `README.md` file there. 

## Contributing

Contributions are welcome! If you'd like to add new replicas or improve existing ones, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add <new replica>'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

Please ensure your contributions adhere to the project's coding standards and include a test if applicable.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
