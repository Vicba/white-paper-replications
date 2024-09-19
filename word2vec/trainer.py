import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from dataloader import get_dataloader_and_vocab
from models import CBOW, SkipGram


class Trainer:
    """Trainer class to handle model training and validation."""

    def __init__(self, model, epochs, train_dataloader, train_steps, val_dataloader, val_steps, criterion, optimizer, scheduler, device):
        self.model = model.to(device)
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.train_steps = train_steps
        self.val_dataloader = val_dataloader
        self.val_steps = val_steps
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.losses = {"train": [], "val": []}

    def train(self):
        for epoch in range(self.epochs):
            train_loss = self._run_epoch(train=True)
            val_loss = self._run_epoch(train=False)
            self.scheduler.step()
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")

    def _run_epoch(self, train=True):
        data_loader = self.train_dataloader if train else self.val_dataloader
        total_steps = self.train_steps if train else self.val_steps
        running_loss = []

        if train:
            self.model.train()
        else:
            self.model.eval()

        for i, (inputs, labels) in enumerate(data_loader, 1):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            if train:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
            running_loss.append(loss.item())
            if i == total_steps:
                break

        avg_loss = np.mean(running_loss)
        if train:
            self.losses["train"].append(avg_loss)
        else:
            self.losses["val"].append(avg_loss)
        return avg_loss

def get_model(model_name, vocab_size, embed_dim):
    if model_name == "cbow":
        return CBOW(vocab_size=vocab_size, embed_dim=embed_dim)
    elif model_name == "skipgram":
        return SkipGram(vocab_size=vocab_size, embed_dim=embed_dim)
    else:
        raise ValueError("Model name must be either 'cbow' or 'skipgram'")

def get_scheduler(optimizer, total_epochs):
    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def main():
    config = {
        "model_name": "cbow",              # Model: 'cbow' or 'skipgram'
        "embed_dim": 300,                  # embedding dimension
        "data_dir": "./data",              # Path to data directory
        "train_batch_size": 32,            # Batch size for training
        "val_batch_size": 32,              # Batch size for validation
        "learning_rate": 0.001,            # Learning rate
        "epochs": 5,                      # Number of epochs
        "train_steps": 100,                # Max steps per training epoch
        "val_steps": 20,                   # Max steps per validation epoch
        "shuffle": True                    # Shuffle the data
    }

    # Load data and vocabulary
    train_dataloader, vocab = get_dataloader_and_vocab(
        model_name=config["model_name"],
        ds_type="train",
        data_dir=config["data_dir"],
        batch_size=config["train_batch_size"],
        shuffle=config["shuffle"]
    )
    val_dataloader, _ = get_dataloader_and_vocab(
        model_name=config["model_name"],
        ds_type="valid",
        data_dir=config["data_dir"],
        batch_size=config["val_batch_size"],
        shuffle=config["shuffle"],
    )
    
    vocab_size = len(vocab.get_stoi())
    print(f"Vocabulary size: {vocab_size}")

    # Initialize model, optimizer, criterion, and scheduler
    model = get_model(config["model_name"], vocab_size, embed_dim=config["embed_dim"])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = get_scheduler(optimizer, config["epochs"])
    criterion = nn.CrossEntropyLoss()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize and run trainer
    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        train_steps=config["train_steps"],
        val_dataloader=val_dataloader,
        val_steps=config["val_steps"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    trainer.train()


if __name__ == '__main__':
    main()
