import torch
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import BertTokenizer
from datasets import load_dataset
from PIL import Image

class ImageTextDataset(Dataset):
    def __init__(self, split='train', transform=None):
        self.dataset = load_dataset('nlphuji/flickr30k', split=split)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = Image.open(self.dataset[idx]['image'])
        text = self.dataset[idx]['caption']
        
        # tokenize caption
        text_tokens = self.tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors="pt")

        if self.transform:
            image = self.transform(image)

        return image, text_tokens['input_ids'].squeeze(0)  # img shape: (3, 224, 224), text shape: (77,)

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ImageTextDataset(split='train', transform=transform)
    print(f"Dataset length: {len(dataset)}")

    for i in range(5):
        image, text_tokens = dataset[i]
        print(f"Sample {i}:")
        print(f"  Image shape: {image.shape}")  # should be (3, 224, 224)
        print(f"  Text tokens shape: {text_tokens.shape}")  # should be (77,)
