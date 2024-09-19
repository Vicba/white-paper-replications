"""
CBOW: predicting target word based on the context words
"""
import torch
import torch.nn as nn
import torch.optim as optim
import re
from stop_words import get_stop_words
import nltk
from nltk.tokenize import word_tokenize

class CBOW(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super(CBOW, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, inputs):
        embeddings = self.embeddings(inputs).mean(1).squeeze()
        return self.fc(embeddings)
    
def preprocess(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z]+", " ", text).strip()
    text = " ".join([word for word in text.split() if word not in get_stop_words('en')])
    return text

def create_dataset(window_size: int):
    with open("./sample_text.txt") as f:
        raw_text = f.read()

    text_cleaned = preprocess(raw_text)
    
    # do very simple tokenization
    tokenized_text = [ token for token in text_cleaned.split()]
    vocab = sorted(set(tokenized_text))

    id2label = {i: word for i, word in enumerate(vocab)}
    label2id = {word: i for i, word in enumerate(vocab)}

    # generate training data with context: window before and after
    data = []
    for i in range(window_size, len(tokenized_text) - window_size):
        context = [
            tokenized_text[i - window_size],
            tokenized_text[i - window_size + 1],
            tokenized_text[i + window_size - 1],
            tokenized_text[i + window_size],
        ]
        target = tokenized_text[i]

        # map context and target to indices and append to data
        context_ids = [label2id[word] for word in context]
        target_id = label2id[target]
        data.append((context_ids, target_id))

    return data, label2id, id2label

def main():
    embed_dim = 300

    data, label2id, id2label = create_dataset(window_size=2)

    model = CBOW(embed_dim=embed_dim, vocab_size=len(label2id))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    context_data, labels = zip(*data)
    context_data = torch.tensor(context_data)
    labels = torch.tensor(labels)

    dataset = torch.utils.data.TensorDataset(context_data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(100):
        for context, label in dataloader:
            pred = model(context)
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch: {epoch}, loss: {loss.item()}")

if __name__ == "__main__":
    main()