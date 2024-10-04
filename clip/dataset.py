import torch
import requests
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import BertTokenizer
from datasets import load_dataset
from PIL import Image
from io import BytesIO

class ImageTextDataset(Dataset):
    def __init__(self, split='test', transform=None):
        self.dataset = load_dataset('nlphuji/flickr30k', split=f'{split}[:1000]') # take sample just for simplicity
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        if self.transform:
            image = self.transform(image)

        text = self.dataset[idx]['caption'][0] 
        
        text_tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors="pt")

        # print(f"raw text tokens shape before squeezing: {text_tokens['input_ids'].shape}")
        # print(image.shape, text_tokens['input_ids'].squeeze(0))
        return image, text_tokens['input_ids'].squeeze(0)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ImageTextDataset(split='test', transform=transform)
    print(f"Dataset length: {len(dataset)}")

    for i in range(3):
        image, text_tokens = dataset[i]
        print(f"Sample {i}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Text tokens shape: {text_tokens.shape}") 
