import torch
import torch.nn as nn
from embedding.bert_embedding import BERTEmbedding
from blocks.encoder_layer import EncoderLayer
from transformers import BertTokenizer, AutoTokenizer
from preprocess import MovieDialogueProcessor
from dataset import BERTDataset
from torch.utils.data import DataLoader

class BERT(nn.Module):
    def __init__(self, d_model, vocab_size, n_layers, n_head, max_seq_len, dropout=0.01):
        super(BERT, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers

        self.d_ff = d_model * 4

        self.embedding = BERTEmbedding(vocab_size, d_model, max_seq_len)

        self.encoder_blocks = nn.ModuleList(
            [EncoderLayer(d_model, n_head, self.d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(self, x, segment_info):
        # attention masking for padded token
        # (batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x
    

class NSP(nn.Module):
  """ binary classification model, is next sentence or not """
  def __init__(self, hidden):
    super(NSP, self).__init__()
    # hidden = bert model output size
    self.fc = nn.Linear(hidden, 2)
    self.softmax = nn.LogSoftmax(dim=-1)

  def forward(self, x):
    # only use first token wich is the CLS token
    return self.softmax(self.fc(x[:, 0]))


class MLM(nn.Module):
  """ n-class, predict origin token fro mmasked input sequence """
  def __init__(self, hidden, vocab_size):
    super(MLM, self).__init__()

    self.fc = nn.Linear(hidden, vocab_size)
    self.softmax = nn.LogSoftmax(dim=-1)

  def forward(self, x):
    return self.softmax(self.fc(x))
  

class BERTLM(nn.Module):
  """ BERT LM, using next sentence prediction and masked token prediction """
  def __init__(self, bert: BERT, vocab_size):
    super(BERTLM, self).__init__()

    self.bert = bert
    self.nsp = NSP(self.bert.d_model)
    self.mlm = MLM(self.bert.d_model, vocab_size)

  def forward(self, x, segment_label):
    x = self.bert(x, segment_label)
    return self.nsp(x), self.mlm(x)



if __name__ == "__main__":
    MAX_LEN = 64
    # tokenizer = BertTokenizer.from_pretrained('./bert-it-1/bert-it-vocab.txt', local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    corpus_movie_conv = './datasets/movie_conversations.txt'
    corpus_movie_lines = './datasets/movie_lines.txt'
    processor = MovieDialogueProcessor(corpus_movie_conv, corpus_movie_lines)
    processor.load_data()
    pairs = processor.get_pairs()

    train_data = BERTDataset(pairs, seq_len=MAX_LEN, tokenizer=tokenizer)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)
    sample_data = next(iter(train_loader))

    ### test
    bert_model = BERT(vocab_size=len(tokenizer.vocab), d_model=768, n_layers=12, n_head=12, max_seq_len=MAX_LEN)
    bert_result = bert_model(sample_data['bert_input'], sample_data['segment_label'])
    print(bert_result.size())
    print()

    bert_lm = BERTLM(bert_model, vocab_size=len(tokenizer.vocab))
    final_result = bert_lm(sample_data['bert_input'], sample_data['segment_label'])

    print("NEXT SENTENCE PREDICTION RESULT")
    print(final_result[0])
    print(final_result[0].size())

    print("MASKED LANGUAGE PREDICTION RESULT")
    print(final_result[1])
    print(final_result[1].size())