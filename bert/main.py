"""
Implement both pretrain and fine-tuning
"""
import argparse
from torch.utils.data import DataLoader
from dataset import RandomTextDataset
from trainer import Trainer
from model import BERT
from vocab import Vocabulary

# constants are of BERT base
n_layers = 12
n_head = 12
embed_dim = 768

# constants BERT large
n_layers = 24
n_head = 16
embed_dim = 1024

# fine-tune 
batch_size = 32
epochs = 3
lr = 5e-5



def main():
    vocab_file_path = './text'
    vocab = Vocabulary(vocab_file_path, max_vocab_size=30522)

    # Prepare the dataset
    train_dataset = RandomTextDataset(vocab, num_samples=1000, max_seq_len=512)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    val_dataset = RandomTextDataset(vocab, num_samples=200, max_seq_len=512)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Initialize the BERT model and trainer
    d_model = 768
    n_head = 12
    d_ff = 3072
    n_layers = 12

    bert_model = BERT(len(vocab), d_model, n_head, d_ff, n_layers, max_seq_len=512)
    trainer = Trainer(bert_model, train_dataloader, val_dataloader, task='pretrain')

    # Start pretraining
    trainer.pretrain(epochs=3)
    trainer.evaluate()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BERT model")
    parser.add_argument("--pretrain", default=True, help="Pretrains the model")
    parser.add_argument("--finetune", default=True, help="Fine-tunes the model")
    parser.add_argument("--bert-size", default="base", help="Fine-tunes the model")

    main(parser)

