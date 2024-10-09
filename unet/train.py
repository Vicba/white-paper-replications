import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from unet import UNet
from dataset import SegmentationDataset

def train_model(model, device, train_loader, optimizer, epochs):
    model.to(device)
    loss_history = []

    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(X)

            loss = F.cross_entropy(output, y)
            loss.backward()
            epoch_loss += loss.item()

            optimizer.step()
            
        average_loss = epoch_loss / len(train_loader)
        loss_history.append(average_loss)
        print(f'Epoch {epoch}, Average loss: {average_loss:.6f}')

    return loss_history

def test_model(model, device, test_loader):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    total_pixels = 0

    with torch.inference_mode():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            output = model(X)
            pred = output.argmax(dim=1)

            test_loss += F.cross_entropy(output, y, reduction='sum').item()
            correct += (pred == y).sum().item()
            total_pixels += y.numel()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / total_pixels
    
    return accuracy

def plot_results(model, dataset, device, num_samples=5):
    model.eval()
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))

    for i in range(num_samples):
        image, true_mask = dataset[np.random.randint(len(dataset))]
        image = image.unsqueeze(0).to(device)

        with torch.inference_mode():
            pred_mask = model(image).argmax(dim=1).squeeze().cpu().numpy()
        
        axs[i, 0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
        axs[i, 0].set_title('Input Image')
        axs[i, 1].imshow(true_mask, cmap='gray')
        axs[i, 1].set_title('True Mask')
        axs[i, 2].imshow(pred_mask, cmap='gray')
        axs[i, 2].set_title('Predicted Mask')
    
    plt.tight_layout()
    plt.show()

def plot_loss_curve(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SegmentationDataset(num_samples=100, img_size=128)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # init model
    model = UNet(n_channels=1, n_classes=2)  # 2 classes: background and object
    optimizer = optim.Adam(model.parameters())

    print("training started...")
    loss_history = train_model(model, device, train_loader, optimizer, epochs=3)

    print("testing...")
    accuracy = test_model(model, device, test_loader)

    plot_loss_curve(loss_history)

    plot_results(model, dataset, device)