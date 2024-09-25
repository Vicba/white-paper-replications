import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import BERT
from dataset import BERTPretrainingDataset

def pretrain_bert(model, dataset, epochs, batch_size, learning_rate, vocab_size):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Loss functions
    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index
    nsp_loss_fn = nn.BCEWithLogitsLoss()  # Binary classification for next sentence prediction

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        total_loss = 0
        
        for input_ids, token_type_ids, attention_mask, mlm_labels, is_next in tqdm(data_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad()

            # Move tensors to the device
            input_ids = input_ids.to(model.device)
            token_type_ids = token_type_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            mlm_labels = mlm_labels.to(model.device)
            is_next = is_next.to(model.device).float()  # Convert to float for BCEWithLogitsLoss

            # Forward pass
            outputs = model(input_ids, token_type_ids, attention_mask)

            # MLM Loss
            mlm_loss = mlm_loss_fn(outputs[0].view(-1, vocab_size), mlm_labels.view(-1))  # Ensure shapes are compatible
            
            # NSP Loss
            nsp_logits = outputs[1]  # Assuming the model returns NSP logits as the second output
            nsp_loss = nsp_loss_fn(nsp_logits.view(-1), is_next)

            # Total loss
            total_loss = mlm_loss + nsp_loss

            total_loss.backward()
            optimizer.step()

            print(f"MLM Loss: {mlm_loss.item():.4f}, NSP Loss: {nsp_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Total Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # Set hyperparameters
    vocab_size = 30522
    d_model = 768
    n_head = 12
    n_layers = 12
    max_seq_len = 512
    batch_size = 32
    learning_rate = 1e-4
    epochs = 1  # Set to 3 for more training

    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create the dataset
    dataset = BERTPretrainingDataset('text.txt', max_seq_len=max_seq_len, tokenizer=tokenizer)

    # Initialize the model
    model = BERT(vocab_size, d_model, n_layers, n_head, max_seq_len)
    model.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
    model.to(model.device)  # Move model to the appropriate device

    # Pretrain the model
    pretrain_bert(model, dataset, epochs, batch_size, learning_rate, vocab_size)
