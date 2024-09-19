import os
import torch
from functools import partial
from torch.utils.data import DataLoader
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
# from torchtext.datasets import WikiText103

from constants import (
    N_WORDS,
    MIN_WORD_FREQUENCY,
    MAX_SEQ_LENGTH,
)

def get_english_tokenizer():
    tokenizer = get_tokenizer("basic_english", language="en")
    return tokenizer

# def get_data_iterator(ds_type, data_dir):
#     data_iter = WikiText103(root=data_dir, split=(ds_type))
#     data_iter = to_map_style_dataset(data_iter)
#     return data_iter

def read_file(file_path):
    """Read the file line by line and yield each line."""
    with open(file_path, 'r') as f:
        for line in f:
            yield line.strip()

def get_data_iterator(ds_type, data_dir):
    """
    Create an iterator for the WikiText dataset.

    Args:
        ds_type (str): The type of dataset split (e.g., 'train', 'valid', 'test').
        data_dir (str): The directory where the dataset is stored.

    Returns:
        Iterator: An iterator over the dataset.
    """
    data_path = os.path.join(data_dir, 'wikitext-103', f'wiki.{ds_type}.tokens')
    data_iter = read_file(data_path)
    data_iter = to_map_style_dataset(data_iter)
    return data_iter

def build_vocab(data_iter, tokenizer):
    vocab = build_vocab_from_iterator(
        map(tokenizer, data_iter),  # Tokenize each item in the data iterator
        specials=["<unk>"],  # Add a special token for unknown words
        min_freq=MIN_WORD_FREQUENCY  # Set minimum frequency for words to be included in the vocab
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab 

def collate_cbow(batch, tokenize_text):
    """
    Collate function for CBOW model used with DataLoader.
    For each word in the dataset, the function creates pairs of context words and target words.

    Args:
        batch: List of text paragraphs, where each paragraph is a string of text.
        tokenize_text: A function that tokenizes and converts text to integer indices using the vocabulary.

    Returns:
        A tuple of two tensors:
        - context_data (torch.Tensor): Tensor containing the context words for each target word.
        - target_data (torch.Tensor): Tensor containing the target words corresponding to each context.

    The tensors have the following shapes:
        - context_data: (num_pairs, N_WORDS * 2) where N_WORDS is the size of the context window.
        - target_data: (num_pairs,) where num_pairs is the number of context-target pairs created.

    The number of pairs (num_pairs) depends on the length of the input paragraphs and the context window size (N_WORDS).
    """
    context_data, target_data = [], []
    
    for paragraph in batch:
        token_ids = tokenize_text(paragraph)[:MAX_SEQ_LENGTH]  # Truncate early if needed
        if len(token_ids) < N_WORDS * 2 + 1: continue  # Skip if too short
        
        for idx in range(N_WORDS, len(token_ids) - N_WORDS):
            context_window = token_ids[idx - N_WORDS: idx] + token_ids[idx + 1: idx + 1 + N_WORDS]
            context_data.append(context_window)
            target_data.append(token_ids[idx])

    return torch.tensor(context_data, dtype=torch.long), torch.tensor(target_data, dtype=torch.long)


def collate_skipgram(batch, tokenize_text):
    """
    Collate function for Skip-Gram model used with DataLoader.
    For each word in the dataset, the function creates pairs of target words and context words.

    Args:
        batch: List of text paragraphs, where each paragraph is a string of text.
        tokenize_text: A function that tokenizes and converts text to integer indices using the vocabulary.

    Returns:
        A tuple of two tensors:
        - target_data (torch.Tensor): Tensor containing the target words for each context.
        - context_data (torch.Tensor): Tensor containing the context words corresponding to each target word.
        
    The tensors have the following shapes:
        - target_data: (num_pairs,)
        - context_data: (num_pairs,)
        
    The number of pairs (num_pairs) depends on the length of the input paragraphs and the context window size (N_WORDS).
    """
    target_data, context_data = [], []
    
    for paragraph in batch:
        token_ids = tokenize_text(paragraph)[:MAX_SEQ_LENGTH]  # Truncate early if needed
        if len(token_ids) < N_WORDS * 2 + 1: continue  # Skip if too short
        
        for idx in range(N_WORDS, len(token_ids) - N_WORDS):
            target_word = token_ids[idx]
            context_window = token_ids[idx - N_WORDS: idx] + token_ids[idx + 1: idx + 1 + N_WORDS]
            for context_word in context_window:
                target_data.append(target_word)
                context_data.append(context_word)

    return torch.tensor(target_data, dtype=torch.long), torch.tensor(context_data, dtype=torch.long)


def get_dataloader_and_vocab(model_name, ds_type, data_dir, batch_size, shuffle):
    data_iter = get_data_iterator(ds_type=ds_type, data_dir=data_dir)
    tokenizer = get_english_tokenizer()

    vocab = build_vocab(data_iter, tokenizer)

    tokenize_text = lambda x: vocab(tokenizer(x))

    collate_fn = collate_cbow if model_name == "cbow" else collate_skipgram

    dataloader = DataLoader(
        data_iter,
        batch_size,
        shuffle,
        collate_fn=partial(collate_fn, tokenize_text=tokenize_text)
    )
    return dataloader, vocab
