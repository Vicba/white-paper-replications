"""
word2vec model types
Pay attention, there is no Softmax activation in the Linear Layer. 
That’s because PyTorch CrossEntropyLoss expects predictions to be raw, unnormalized scores.
"""
import torch.nn as nn

class CBOW(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_norm: int = 1):
        super(CBOW, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim, max_norm)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, inputs): # model inputs are word ids
        x = self.embeddings(inputs) # Shape: (batch_size, context_size, embed_dim)
        # mean vector represents overall context of input words
        x = x.mean(axis=1) # Shape: (batch_size, embed_dim) - averaging over context words
        x = self.fc(x) # Shape: (batch_size, vocab_size) - output scores for each word
        return x
    

class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_norm: int = 1):
        super(SkipGram, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim, max_norm)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, inputs): # model inputs are word ids
        x = self.embeddings(inputs) # Shape: (batch_size, embed_dim)
        x = self.fc(x) # Shape: (batch_size, vocab_size) - output scores for each word
        return x