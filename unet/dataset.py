import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class SegmentationDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=128):
        self.num_samples = num_samples
        self.img_size = img_size
        self.images, self.masks = self.generate_dataset()

    def generate_dataset(self):
        images = []
        masks = []
        for _ in range(self.num_samples):
            # random background
            image = np.random.rand(self.img_size, self.img_size)
            
            # random shape (circle or rectangle)
            mask = np.zeros((self.img_size, self.img_size))
            shape_type = np.random.choice(['circle', 'rectangle'])
            
            center_x, center_y = np.random.randint(0, self.img_size, 2)
            size = np.random.randint(10, self.img_size // 4)
            
            if shape_type == 'circle':
                y, x = np.ogrid[:self.img_size, :self.img_size]
                dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                mask[dist_from_center <= size] = 1
                
            else:  # rectangle
                x1, y1 = max(0, center_x - size), max(0, center_y - size)
                x2, y2 = min(self.img_size, center_x + size), min(self.img_size, center_y + size)
                mask[y1:y2, x1:x2] = 1
            
            # add shape to img
            image[mask == 1] = 1
            
            images.append(image)
            masks.append(mask)
        
        return np.array(images), np.array(masks)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.FloatTensor(self.images[idx]).unsqueeze(0)  # add channel dim
        mask = torch.LongTensor(self.masks[idx])
        return image, mask


if __name__ == "__main__":
    dataset = SegmentationDataset(num_samples=100, img_size=128)
    
    print(f"SegmentationDataset Test:")
    print(f"Dataset size: {len(dataset)}")
    
    # random item
    idx = np.random.randint(len(dataset))
    image, mask = dataset[idx]
    
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Image min: {image.min().item():.4f}, max: {image.max().item():.4f}")
    print(f"Unique values in mask: {torch.unique(mask)}")
    
    # vizualize img and mask
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image.squeeze(), cmap='gray')
    ax1.set_title("Image")
    ax2.imshow(mask, cmap='gray')
    ax2.set_title("Mask")
    plt.show()
    
    print("ds created successfully!")