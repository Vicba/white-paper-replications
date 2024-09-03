import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader, Dataset
from basic_moe import TransformerEncoderLayerWithMoE
from collections import Counter
from itertools import chain

class Model(nn.Module):
    """
    A simple Transformer model with Mixture of Experts (MoE).
    This model includes an embedding layer, positional encoding, multiple Transformer layers with MoE, and an output layer.
    """
    def __init__(self, model_dim, num_heads, num_experts, hidden_dim, num_layers, vocab_size, max_seq_len, dropout=0.1, top_k=2):
        super(Model, self).__init__()
        # Initialize embedding layer for input tokens
        self.embedding = nn.Embedding(vocab_size, model_dim)
        # Positional encoding to give the model information about the position of tokens in the sequence
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, model_dim))
        # Create a list of Transformer layers with MoE
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerWithMoE(model_dim, num_heads, num_experts, hidden_dim, dropout, top_k)
            for _ in range(num_layers)
        ])
        # Output layer to map the final hidden states to vocabulary size
        self.output_layer = nn.Linear(model_dim, vocab_size)

    def forward(self, src):
        # Add positional encoding to the embedded tokens
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        # Pass through each Transformer layer with MoE
        for layer in self.encoder_layers:
            src = layer(src)
        # Output logits for each token in the sequence
        output = self.output_layer(src)
        return output

class TextDataset(Dataset):
    """
    Custom Dataset class for text classification.
    Converts text data into tensor format and pads/truncates sequences to max_seq_len.
    """
    def __init__(self, data, vocab, max_seq_len):
        self.data = data
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        # Convert text tokens to numerical indices using the vocab
        text = [self.vocab[token] for token in text]
        # Pad or truncate the sequence to the maximum length
        text = text[:self.max_seq_len] + [self.vocab['<pad>']] * (self.max_seq_len - len(text))
        return torch.tensor(text), torch.tensor(label)

def build_vocab(train_data, test_data):
    """
    Build vocabulary from training and test data.
    This function creates a vocabulary that includes special tokens for unknown and padding.
    """
    tokenizer = get_tokenizer('basic_english')
    counter = Counter()  # count token occurrences
    for text, _ in chain(train_data, test_data):
        counter.update(tokenizer(text))  # Update the counter with tokens from both datasets
    # Create vocab with space and <pad> special tokens
    vocab = torchtext.vocab.vocab(counter, specials=[' ', '<pad>'])
    vocab.set_default_index(vocab[' '])  # Set default index for unknown tokens
    return vocab

# EXAMPLE VOCAB
# {
#     '<unk>': 0,
#     '<pad>': 1,
#     'I': 2,
#     'love': 3,
#     'machine': 4,
#     'learning': 5,
#     'is': 6,
#     'great': 7,
#     'enjoy': 8,
#     'fascinating': 9
# }

def process_data(data):
    """
    Process raw text data into token indices and labels.
    This function tokenizes the text and assigns binary labels based on sentiment.
    """
    tokenizer = get_tokenizer('basic_english')  # Initialize tokenizer for basic English
    # (imdb dataset is about pos or neg sentiment)
    return [(tokenizer(text), 1 if label == 'pos' else 0) for text, label in data]  # Tokenize and label data

def preprocess_data(max_seq_len):
    """
    Preprocess the IMDB dataset, including tokenization, vocabulary creation, and DataLoader preparation.
    This function loads the dataset, builds the vocabulary, processes the data, and prepares DataLoaders.
    """
    train_data, test_data = IMDB(split=('train', 'test'))
    
    vocab = build_vocab(train_data, test_data)

    # Process data into token indices and labels
    train_data_processed = process_data(train_data)
    test_data_processed = process_data(test_data)

    train_dataset = TextDataset(train_data_processed, vocab, max_seq_len)
    test_dataset = TextDataset(test_data_processed, vocab, max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, vocab

def train_model(model, train_loader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    This function performs forward and backward passes and updates the model weights.
    """
    model.train()
    total_loss = 0
    for i, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)

        loss = criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)

if __name__ == "__main__":
    # Hyperparameters
    model_dim = 128
    num_heads = 4
    num_experts = 4
    hidden_dim = 512
    num_layers = 2
    max_seq_len = 50  # Maximum seq length for padding
    dropout = 0.1
    top_k = 2  # Top-k experts to use in MoE
    num_epochs = 5
    learning_rate = 0.001

    train_loader, test_loader, vocab = preprocess_data(max_seq_len)
    vocab_size = len(vocab)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(model_dim, num_heads, num_experts, hidden_dim, num_layers, vocab_size, max_seq_len, dropout, top_k).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>']) # Loss function with padding ignored
    # can also ignore by using -100, see param: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        avg_loss = train_model(model, train_loader, optimizer, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print("Training complete.") 
