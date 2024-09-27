import os
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer
from tokenizers import BertWordPieceTokenizer

class CustomTokenizer:
    def __init__(self, vocab_size=30000):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.vocab_size = vocab_size
        self.word_freq = defaultdict(int)
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.splits = {}

    def build_word_frequency(self, corpus):
        for doc in corpus:
            words_with_offsets = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(doc)
            new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                self.word_freq[word] += 1

    def create_alphabet(self):
        alphabet = []
        for word in self.word_freq.keys():
            if word[0] not in alphabet:
                alphabet.append(word[0])
            for letter in word[1:]:
                if f"##{letter}" not in alphabet:
                    alphabet.append(f"##{letter}")
        alphabet.sort()
        return alphabet

    def initialize_splits(self):
        alphabet = self.create_alphabet()
        self.splits = {word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)] for word in self.word_freq.keys()}
        self.vocab += alphabet

    def compute_pair_scores(self):
        letter_freq = defaultdict(int)
        pair_freq = defaultdict(int)

        for word, freq in self.word_freq.items():
            split = self.splits[word]
            if len(split) == 1:
                letter_freq[split[0]] += freq
                continue

            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                letter_freq[split[i]] += freq
                pair_freq[pair] += freq

            letter_freq[split[-1]] += freq

        scores = {
            pair: freq / (letter_freq[pair[0]] * letter_freq[pair[1]])
            for pair, freq in pair_freq.items()
        }
        return scores

    def merge_pair(self, a, b):
        for word in self.word_freq:
            split = self.splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    merge = a + b[2:] if b.startswith("##") else a + b
                    split = split[:i] + [merge] + split[i + 2:]
                else:
                    i += 1
            self.splits[word] = split
        return self.splits

    def train_tokenizer(self, corpus):
        self.build_word_frequency(corpus)
        self.initialize_splits()

        while len(self.vocab) < self.vocab_size:
            scores = self.compute_pair_scores()
            best_pair = max(scores, key=scores.get)
            self.splits = self.merge_pair(*best_pair)
            new_token = (
                best_pair[0] + best_pair[1][2:] if best_pair[1].startswith("##")
                else best_pair[0] + best_pair[1]
            )
            self.vocab.append(new_token)

        print(f'Final Vocab: {self.vocab}')

    def encode_word(self, word):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocab:
                i -= 1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens

    def save_tokenizer(self, paths):
        tokenizer = BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=False,
            strip_accents=False,
            lowercase=True
        )
        tokenizer.train(
            files=paths,
            vocab_size=self.vocab_size,
            min_frequency=5,
            limit_alphabet=1000,
            wordpieces_prefix='##',
            special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
        )
        os.mkdir('./bert-it-1')
        tokenizer.save_model('./bert-it-1', 'bert-it')

    def load_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained('./bert-it-1/bert-it-vocab.txt', local_files_only=True)
        return tokenizer


if __name__ == "__main__":
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    tokenizer = CustomTokenizer()
    tokenizer.train_tokenizer(corpus)

    # Example usage
    print(tokenizer.encode_word("Hugging"))
    print(tokenizer.encode_word("HOgging"))

    # Load additional datasets and train the tokenizer
    # Assuming you have pairs from your dataset processing
    # tokenizer.save_tokenizer(paths)
    # tokenizer = tokenizer.load_tokenizer()
