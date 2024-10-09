import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from unet import UNet
from dataset import SegmentationDataset

def train_model(model, device, train_loader, optimizer, epochs):
    model.to(device)
    loss_history = []

    for epoch in tqdm(range(epochs)):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            loss = F.cross_entropy(output, target)
            loss.backward()
            epoch_loss += loss.item()

            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                
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
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.argmax(dim=1)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            correct += (pred == target).sum().item()
            total_pixels += target.numel()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / total_pixels

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{total_pixels} '
          f'({accuracy:.2f}%)\n')
    
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

    dataset = SegmentationDataset(num_samples=1000, img_size=128)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # init model
    model = UNet(n_channels=1, n_classes=2)  # 2 classes: background and object
    optimizer = optim.Adam(model.parameters())

    print("training started...")
    loss_history = train_model(model, device, train_loader, optimizer, epochs=10)

    print("testing...")
    accuracy = test_model(model, device, test_loader)

    plot_loss_curve(loss_history)

    plot_results(model, dataset, device)