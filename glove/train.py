import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from model import GloVe

WINDOW_SIZE = 2
EMBED_DIM = 50

def tokenize_corpus(corpus):
    return [sentence.lower().split() for sentence in corpus]

def build_vocab(tokenized_corpus):
    """
    Args:
        tokenized_corpus (list of list of str): Tokenized text corpus.
    Returns:
        dict: Vocabulary mapping words to indices.
        Counter: Word counts in the corpus.
    """
    word_counts = Counter([word for sentence in tokenized_corpus for word in sentence])
    vocab = {word: i for i, word in enumerate(word_counts.keys())}
    return vocab, word_counts


def build_cooccurrence_matrix(vocab, tokenized_corpus, window_size):
    """
    Builds a word co-occurrence matrix based on the corpus and window size.
    Args:
        vocab (dict): Vocabulary mapping words to indices.
        tokenized_corpus (list of list of str): Tokenized text corpus.
        window_size (int): The context window size.
    Returns:
        np.ndarray: Co-occurrence matrix.
    """
    vocab_size = len(vocab)
    cooccurrence_matrix = np.zeros((vocab_size, vocab_size))
    # Iterate and count co-occurrences
    for sentence in tokenized_corpus:
        for i, word in enumerate(sentence):
            word_id = vocab[word]
            # Define context window
            start = max(0, i - window_size)
            end = min(len(sentence), i + window_size + 1)
            # Update co-occurrence counts within the window
            for j in range(start, end):
                if i != j:
                    context_word_id = vocab[sentence[j]]
                    cooccurrence_matrix[word_id, context_word_id] += 1

    return cooccurrence_matrix


def weighting_function(x, x_max=100, alpha=0.75):
    """
    The GloVe weighting function to scale co-occurrence counts.
    Args:
        x (float): Co-occurrence value.
        x_max (int): Maximum co-occurrence count for scaling.
        alpha (float): Exponent for scaling.
    Returns:
        float: Weight applied to the co-occurrence count.
    """
    return (x / x_max) ** alpha if x < x_max else 1


def glove_loss(pred, log_x_ij, weight):
    """
    Calculates the GloVe loss function.
    Args:
        pred (Tensor): Predicted log co-occurrence values.
        log_x_ij (Tensor): Actual log of co-occurrence counts.
        weight (Tensor): Weights from the weighting function.
    Returns:
        Tensor: The GloVe loss.
    """
    return torch.mean(weight * (pred - log_x_ij) ** 2)


def get_word_embeddings(model):
    """
    Returns the final word embeddings by averaging word and context embeddings.
    Args:
        model (GloVe): Trained GloVe model.
    Returns:
        Tensor: Final word embeddings.
    """
    return (model.wi.weight + model.wj.weight) / 2


def main():
    corpus = ["I love deep learning", "Deep learning is fun", "I love learning"]
    tokenized_corpus = tokenize_corpus(corpus)

    vocab, word_counts = build_vocab(tokenized_corpus)
    print("vocab: ", vocab)
    vocab_size = len(vocab)

    cooccurrence_matrix = build_cooccurrence_matrix(vocab, tokenized_corpus, WINDOW_SIZE)

    # Apply the weighting function to each element in the co-occurrence matrix
    # np.vectorize creates a vectorized version of the weighting_function, allowing it to be applied element-wise
    # this means that each element in the matrix is passed to the weighting_function without the need for loops
    # https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html
    weight_matrix = np.vectorize(weighting_function)(cooccurrence_matrix)

    # get indices of non-zero elements in the co-occurrence matrix
    i_indices, j_indices = cooccurrence_matrix.nonzero()
    # get co-occurrence values corresponding to the non-zero indices
    cooccurrence_values = cooccurrence_matrix[i_indices, j_indices]
    # convert co-occurrence values to log scale for better numerical stability
    log_cooccurrence = np.log(cooccurrence_values + 1e-10)

    # convert the indices for model input
    i_indices = torch.LongTensor(i_indices)
    j_indices = torch.LongTensor(j_indices)
    # convert log co-occurrence values for loss calculation
    log_cooccurrence = torch.FloatTensor(log_cooccurrence)
    # get weights corresponding to the non-zero indices from the weight matrix
    weights = torch.FloatTensor(weight_matrix[i_indices, j_indices])

    model = GloVe(vocab_size, EMBED_DIM)
    optimizer = optim.SparseAdam(model.parameters(), lr=0.001)

    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        predictions = model(i_indices, j_indices)

        loss = glove_loss(predictions, log_cooccurrence, weights)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")


    word_embeddings = get_word_embeddings(model)
    print(word_embeddings)
    print("Word Embeddings Shape:", word_embeddings.shape)


if __name__ == "__main__":
    main()
