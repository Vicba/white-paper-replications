import torch
import random
import numpy as np
from torch.utils.data import Dataset
from vocab import Vocabulary

class RandomTextDataset(Dataset):
    def __init__(self, vocab, num_samples, max_seq_len=512):
        self.vocab = vocab
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len

        # Generate random text samples
        self.samples = self.generate_random_samples(num_samples)

    def generate_random_samples(self, num_samples):
        samples = []
        for _ in range(num_samples):
            # Randomly generate sentence length
            seq_length = random.randint(5, self.max_seq_len)
            # Create a random sequence of token IDs
            input_ids = np.random.randint(0, len(self.vocab), size=seq_length).tolist()

            # Generate segment IDs
            segment_ids = [0] * seq_length
            
            # Create attention mask
            attention_mask = [1] * seq_length
            
            # Create MLM labels
            mlm_labels = input_ids.copy()
            num_to_mask = int(0.15 * seq_length)
            mask_indices = random.sample(range(seq_length), num_to_mask)
            for idx in mask_indices:
                mlm_labels[idx] = self.vocab.token_to_id['[UNK]']  # Using [UNK] for masked tokens

            # Create NSP labels
            nsp_labels = random.randint(0, 1)

            samples.append((input_ids, segment_ids, attention_mask, mlm_labels, nsp_labels))

        return samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids, segment_ids, attention_mask, mlm_labels, nsp_labels = self.samples[idx]
        
        # Pad sequences to max_seq_len
        input_ids += [self.vocab.token_to_id['[PAD]']] * (self.max_seq_len - len(input_ids))
        segment_ids += [0] * (self.max_seq_len - len(segment_ids))
        attention_mask += [0] * (self.max_seq_len - len(attention_mask))
        mlm_labels += [-100] * (self.max_seq_len - len(mlm_labels))  # -100 will be ignored in loss

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(segment_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(mlm_labels, dtype=torch.long),
            torch.tensor(nsp_labels, dtype=torch.long)
        )
