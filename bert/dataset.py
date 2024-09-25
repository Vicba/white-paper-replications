import torch
import random
from torch.utils.data import Dataset
from transformers import BertTokenizer

class BERTPretrainingDataset(Dataset):
    def __init__(self, file_path, max_seq_len=512, tokenizer=None):
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_len = max_seq_len
        
        # Load text data from the specified file
        with open(file_path, 'r', encoding='utf-8') as f:
            self.texts = f.readlines()
        self.texts = [text.strip() for text in self.texts if text.strip()]  # Remove empty lines

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_a = self.texts[idx]
        
        # Randomly choose to create a next sentence pair or not
        if random.random() < 0.5:
            # Choose a random next sentence from the same corpus
            text_b = random.choice(self.texts)
            is_next = 1  # Positive pair
        else:
            # Choose a random sentence from a different part of the corpus
            text_b = random.choice(self.texts)
            while text_b == text_a:
                text_b = random.choice(self.texts)
            is_next = 0  # Negative pair

        # Encode the pair
        encoded_pair = self.tokenizer.encode_plus(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Prepare input tensors
        input_ids = encoded_pair['input_ids'].squeeze(0)
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # Segment IDs
        attention_mask = encoded_pair['attention_mask'].squeeze(0)  # Attention Mask

        # Add a new dimension for compatibility
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_length)

        # Create labels for MLM
        mlm_labels = input_ids.clone()
        # Randomly mask some of the tokens
        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < 0.15) * (input_ids != 0)  # 15% tokens will be masked
        for i in range(input_ids.shape[0]):
            if mask_arr[i]:
                # 80% of the time, replace with [MASK]
                if random.random() < 0.8:
                    mlm_labels[i] = self.tokenizer.mask_token_id
                # 10% of the time, keep original
                elif random.random() < 0.5:
                    mlm_labels[i] = input_ids[i]
                # 10% of the time, replace with random token
                else:
                    mlm_labels[i] = random.randint(1, self.tokenizer.vocab_size - 1)

        return input_ids, token_type_ids, attention_mask, mlm_labels, is_next

# Example usage:
if __name__ == "__main__":
    file_path = 'text.txt'
    dataset = BERTPretrainingDataset(file_path)

    input_ids, token_type_ids, attention_mask, mlm_labels, is_next = dataset[0]
    decoded_text = dataset.tokenizer.decode(input_ids, skip_special_tokens=True)

    print("Input IDs:", input_ids)
    print("Token Type IDs:", token_type_ids)
    print("Attention Mask:", attention_mask)
    print("MLM Labels:", mlm_labels)
    print("Is Next:", is_next)
    print("Decoded Text:", decoded_text)  # This will show the decoded text
    print(attention_mask.shape)  # Verify the shape
