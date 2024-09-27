import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import itertools
from transformers import AutoTokenizer
from preprocess import MovieDialogueProcessor

class BERTDataset(Dataset):
    def __init__(self, data_pair, tokenizer, seq_len=64):

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_lines = len(data_pair)
        self.lines = data_pair

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):

        # Step 1: get random sentence pair, either negative or positive (saved as is_next)
        sent1, sent2, is_next = self.get_sent(item)

        # Step 2: replace random words in sentence with mask / random words
        sent1_random, sent1_label = self.random_word(sent1)
        sent2_random, sent2_label = self.random_word(sent2)

        # Step 3: Adding CLS and SEP tokens to the start and end of sentences
        # Adding PAD token for labels
        sent1 = [self.tokenizer.vocab['[CLS]']] + sent1_random + [self.tokenizer.vocab['[SEP]']]
        sent2 = sent2_random + [self.tokenizer.vocab['[SEP]']]
        sent1_label = [self.tokenizer.vocab['[PAD]']] + sent1_label + [self.tokenizer.vocab['[PAD]']]
        sent2_label = sent2_label + [self.tokenizer.vocab['[PAD]']]

        # Step 4: combine sentence 1 and 2 as one input
        # adding PAD tokens to make the sentence same length as seq_len
        segment_label = ([1 for _ in range(len(sent1))] + [2 for _ in range(len(sent2))])[:self.seq_len]
        bert_input = (sent1 + sent2)[:self.seq_len]
        bert_label = (sent1_label + sent2_label)[:self.seq_len]
        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next}

        return {k: torch.tensor(v) for k, v in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        output = []

        # 15% of the tokens would be replaced
        for i, token in enumerate(tokens):
            prob = random.random()

            # remove cls and sep token
            token_id = self.tokenizer(token)['input_ids'][1:-1]

            # 15% chance of altering token
            if prob < 0.15:
                prob /= 0.15

                # 80% chance change token to mask token
                if prob < 0.8:
                    for i in range(len(token_id)):
                        output.append(self.tokenizer.vocab['[MASK]'])

                # 10% chance change token to random token
                elif prob < 0.9:
                    for i in range(len(token_id)):
                        output.append(random.randrange(len(self.tokenizer.vocab)))

                # 10% chance change token to current token
                else:
                    output.append(token_id)

                output_label.append(token_id)

            else:
                output.append(token_id)
                for i in range(len(token_id)):
                    output_label.append(0)

        # flattening
        output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
        output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
        assert len(output) == len(output_label)
        return output, output_label

    def get_sent(self, index):
        '''return random sentence pair'''
        sent1, sent2 = self.get_corpus_line(index)

        # negative or positive pair, for next sentence prediction
        if random.random() > 0.5:
            return sent1, sent2, 1
        else:
            return sent1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        '''return sentence pair'''
        return self.lines[item][0], self.lines[item][1]

    def get_random_line(self):
        '''return random single sentence'''
        return self.lines[random.randrange(len(self.lines))][1]
    

if __name__ == "__main__":
    MAX_LEN = 64
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    corpus_movie_conv = './datasets/movie_conversations.txt'
    corpus_movie_lines = './datasets/movie_lines.txt'
    processor = MovieDialogueProcessor(corpus_movie_conv, corpus_movie_lines)
    processor.load_data()
    pairs = processor.get_pairs()

    print("\n")
    train_data = BERTDataset(pairs, seq_len=MAX_LEN, tokenizer=tokenizer)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)
    sample_data = next(iter(train_loader))
    print('Batch Size', sample_data['bert_input'].size())

    result = train_data[random.randrange(len(train_data))]
    print(result)