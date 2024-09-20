import torch
import torch.nn as nn

class GloVe(nn.Module):
    def __init__(self, vocab_size, embed_dim, sparse=True):
        """
        Initializes the word and context embeddings and biases.
        
        Args:
            vocab_size (int): The size of the vocabulary.
            embed_dim (int): The dimension of the embeddings.
        """
        super(GloVe, self).__init__()
        # Using sparse gradients for memory efficiency, especially when working with large vocabularies.

        # Word embeddings and their biases
        self.wi = nn.Embedding(vocab_size, embed_dim, sparse=sparse)
        self.bi = nn.Embedding(vocab_size, 1, sparse=sparse)

        # context embeddings and their biases
        # Naming convention: sometimes "tilde" (~) is used to denote context embeddings.
        self.wj = nn.Embedding(vocab_size, embed_dim, sparse=sparse)
        self.bj = nn.Embedding(vocab_size, 1, sparse=sparse)

        # Xavier Uniform initialization: ensures weights are sampled from a distribution
        # with controlled variance, avoiding vanishing or exploding gradients.
        # https://pytorch.org/docs/stable/nn.init.html
        nn.init.xavier_uniform_(self.wi.weight)
        nn.init.xavier_uniform_(self.wj.weight)

        # Bias terms: initialized to zero, allowing the model to learn them during training.
        # https://pytorch.org/docs/stable/nn.init.html
        nn.init.zeros_(self.bi.weight)
        nn.init.zeros_(self.bj.weight)

        # Custom Xavier initialization function can also be used.
        # Commenting it out here and comment the above xavier initizlization.
        # self.custom_xavier_init(vocab_size, embed_dim)

    def custom_xavier_init(self, vocab_size, embed_dim):
        """
        Custom Xavier initialization: a more flexible approach to initializing embeddings.
        """
        initrange = (2.0 / (vocab_size + embed_dim))**0.5
        self.wi.weight.data.uniform_(-initrange, initrange)
        self.wj.weight.data.uniform_(-initrange, initrange)

    def forward(self, i_idx, j_idx):
        """
        Args:
            i_idx (Tensor): Indices of the target words.
            j_idx (Tensor): Indices of the context words.
        Returns:
            Tensor: Predicted values based on word and context embeddings.
        """
        # get word and context embeddings given word indices.
        w_i = self.wi(i_idx)  # Target word embeddings
        w_j = self.wj(j_idx)  # Context word embeddings

        # get bias terms for the target and context words.
        b_i = self.bi(i_idx).squeeze()  # Target word bias
        b_j = self.bj(j_idx).squeeze()  # Context word bias

        return torch.matmul(w_i, w_j.T) + b_i + b_j
