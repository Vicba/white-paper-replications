import os
import torch
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from model import CLIPModel
from utils.clip_loss import CLIPLoss
from dataset import ImageTextDataset

def train_clip(model, dataloader, optimizer, num_epochs=10):
    model.train()
    loss_fn = CLIPLoss(temperature=0.1)
    
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        for i, (images, text_tokens) in tqdm(enumerate(dataloader)):     
            print("epoch:", epoch, "iteration:", i)       
            optimizer.zero_grad()
            
            image_embeddings, text_embeddings = model(images, text_tokens)

            loss = loss_fn(image_embeddings, text_embeddings)
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")


if __name__ == "__main__":    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = ImageTextDataset(split='test', transform=transform) # bc just for demo purposes
    subset_dataset = Subset(dataset, list(range(100)))

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=os.cpu_count())

    model = CLIPModel()
    optimizer = Adam(model.parameters(), lr=1e-4)

    print("start training...")
    train_clip(model, dataloader, optimizer, num_epochs=3)
