import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import CLIPModel
from utils.clip_loss import CLIPLoss
from dataset import ImageTextDataset

def train_clip(model, dataloader, optimizer, num_epochs=10):
    model.train()
    loss_fn = CLIPLoss(temperature=0.1)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for (images, text_tokens) in dataloader:            
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
    
    dataset = ImageTextDataset(split='train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = CLIPModel()
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    train_clip(model, dataloader, optimizer, num_epochs=10)
