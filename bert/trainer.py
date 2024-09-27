import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from preprocess import MovieDialogueProcessor
from dataset import BERTDataset
from torch.utils.data import DataLoader
from model import BERT, BERTLM

class ScheduledOptim():
    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class BERTTrainer():
    def __init__(
        self,
        model,
        train_dataloader,
        test_dataloader=None,
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.0, 0.999),
        warmup_steps=10000,
        log_freq=5,
        device="cuda"
    ):
        self.device = device
        self.model = model
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # or just AdamW
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(
            self.optim, self.model.bert.d_model, n_warmup_steps=warmup_steps)

        # Negative Log Likelihood Loss for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)
        self.log_freq = log_freq
        print("total # parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self._run_epoch(epoch, self.train_data, mode="train")

    def test(self, epoch):
        self._run_epoch(epoch, self.test_data, mode="test")

    def _run_epoch(self, epoch, data_loader, mode):
        avg_loss = 0.0
        total_corr, total_elem = 0, 0

        for i, data in tqdm(enumerate(data_loader)):
            # 0. Move batch data to the device (GPU or CPU)
            data = {k: v.to(self.device) for k, v in data.items()}

            # 1. Forward pass for nsp and mlm
            nsp_output, mlm_output = self.model(data["bert_input"], data["segment_label"])

            # 2-1. Compute NLL loss for is_next classification
            nsp_loss = self.criterion(nsp_output, data["is_next"])

            # 2-2. Compute NLL loss for predicting masked tokens
            mlm_loss = self.criterion(mlm_output.transpose(1, 2), data["bert_label"])

            # 2-3. Combine losses for final loss
            loss = nsp_loss + mlm_loss

            # 3. Backward pass and optimization in training mode
            if mode == "train":
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # Calculate accuracy for nsp
            correct = nsp_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            total_corr += correct
            total_elem += data["is_next"].nelement()

            # Logging progress at every `log_freq` iteration
            if i % self.log_freq == 0:
                print({
                    "epoch": epoch,
                    "avg_loss": avg_loss / (i + 1),
                    "avg_acc": total_corr / total_elem * 100,
                    "loss": loss.item()
                })

        # Print final stats for the epoch
        print(
            f"EP{epoch}, {mode}: avg_loss={avg_loss / len(data_loader)}, total_acc={total_corr * 100.0 / total_elem}"
        )

if __name__ == "__main__":
    # reduce params to run on colab
    subset_size = 100  # original: 1000, adjust the size if needed
    MAX_LEN = 32 # original: 64, change to smaller value if needed
    batch_size = 16 # original: 32, change to smaller if needed but affects convergence model (require more iter)

    corpus_movie_conv = './datasets/movie_conversations.txt'
    corpus_movie_lines = './datasets/movie_lines.txt'
    processor = MovieDialogueProcessor(corpus_movie_conv, corpus_movie_lines)
    processor.load_data()
    pairs = processor.get_pairs()
    pairs_subset = pairs[:subset_size]

    # tokenizer = BertTokenizer.from_pretrained('./bert-it-1/bert-it-vocab.txt', local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


    train_data = BERTDataset(pairs_subset, seq_len=MAX_LEN, tokenizer=tokenizer)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)

    bert_model = BERT(vocab_size=len(tokenizer.vocab), d_model=768, n_layers=12, n_head=12, max_seq_len=MAX_LEN)

    bert_lm = BERTLM(bert=bert_model, vocab_size=len(tokenizer.vocab))

    trainer = BERTTrainer(bert_lm, train_loader, device='cpu')

    epochs = 5

    for epoch in range(epochs):
        trainer.train(epoch)