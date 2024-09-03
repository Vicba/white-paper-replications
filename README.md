# White Paper Replicas

Welcome to the **White Paper Replications** repository! This project is dedicated to replicating state-of-the-art AI research from influential white papers, primarily using PyTorch and Python.

`Disclaimer:` The code is written by me, so it's not perfect. I'm still learning and trying to understand the code. It's a combination of parts I found in different resources.

Planning on adding these in the future
- RoPE
- FlashAttention

## Overview

In this repository, you'll find implementations of various AI models and algorithms as described in prominent research papers. The goal is to better understand them by coding and trying to make them as intuitive as possible.

## Table of Contents

- [Getting Started](#getting-started)
- [Available Replicas](#available-replicas)
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

## Available Replicas

Here's a list of the white papers and their corresponding implementations available in this repository:

- **["Attention is all you need" paper](https://arxiv.org/abs/1706.03762)**: Introduces the Transformer model, which uses self-attention to focus on important words in a sentence, making it faster and better at understanding long sentences compared to older models.
- **["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929)**: Visual transformer (ViT) is a type of neural network that uses self-attention to process images for classification tasks.

Each implementation is located in its own directory, which includes:

- **`README.md`**: Detailed instructions on how to use the code for that specific paper.
<!-- - **`main.py`**: The main script to run the implementation. -->
- **`requirements.txt`**: Python dependencies required for that implementation.
<!-- - **`data/`**: Dataset and other necessary files (if applicable). -->

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
4. Commit your changes (`git commit -am 'Add new replica'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

Please ensure your contributions adhere to the project's coding standards and include a test if applicable.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.