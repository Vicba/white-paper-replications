import torch
import torch.optim as optim

class Trainer:
    def __init__(self, model, train_loader, val_loader, task='pretrain'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task = task
        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)

    def pretrain(self, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in self.train_loader:
                input_ids, segment_ids, attention_mask, mlm_labels, nsp_labels = batch
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, segment_ids, attention_mask, mlm_labels, nsp_labels)
                loss = outputs[0]  # Assume the model returns loss first
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(self.train_loader)}")

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids, segment_ids, attention_mask, mlm_labels, nsp_labels = batch
                outputs = self.model(input_ids, segment_ids, attention_mask, mlm_labels, nsp_labels)
                loss = outputs[0]
                total_loss += loss.item()
        print(f"Validation Loss: {total_loss / len(self.val_loader)}")
