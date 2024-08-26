import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from model.vit import ViT
from tqdm.auto import tqdm

def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, target) in tqdm(enumerate(train_loader), leave=False):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{epochs} completed')
        print(f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%')

def predict(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for data, target in tqdm(test_loader):
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')

if __name__ == '__main__':
    # Load and preprocess data
    transform = transforms.Compose([
        transforms.Resize((32, 32)), # very small 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Use a smaller subset of the data
    train_subset = Subset(train_dataset, range(5000))  # Use only 5000 training images
    test_subset = Subset(test_dataset, range(1000))    # Use only 1000 test images

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=os.cpu_count())
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=os.cpu_count())

    # Initialize the ViT model
    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=256,
        n_layers=6,
        heads=8,
        mlp_dim=512
    )

    # Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, train_loader, criterion, optimizer, epochs=10)

    # Make predictions
    predict(model, test_loader)

    # save model
    torch.save(model.state_dict(), 'vit_cifar10.pth')