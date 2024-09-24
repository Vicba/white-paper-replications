import re
from collections import defaultdict, Counter

class Vocab:
    def __init__(self, file_path, max_vocab_size=30000):
        self.label2id = {}
        self.id2label = {}
        self.max_vocab_size = max_vocab_size
        self.special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]']
        self.build_vocab(file_path)

    def build_vocab(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Preprocess the text
        text = text.lower()
        text = re.sub(r'[^a-z\s]+', '', text).strip()
        
        # Split text into words and count frequency
        words = text.split()
        word_freq = Counter(words)

        # Initialize vocabulary and add special tokens
        self.label2id = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.id2label = {idx: token for idx, token in enumerate(self.special_tokens)}

        # Tokenize and build subword vocabulary
        subword_vocab = self.wordpiece_tokenize(word_freq)

        # Add subwords to vocab
        for subword in subword_vocab:
            if len(self.label2id) < self.max_vocab_size:
                self.label2id[subword] = len(self.label2id)
                self.id2label[len(self.label2id) - 1] = subword

    def wordpiece_tokenize(self, word_freq):
        """Tokenizes words into subwords using the WordPiece method."""
        subword_vocab = defaultdict(int)

        for word, freq in word_freq.items():
            # Initialize subword with the entire word and count its frequency
            subword_vocab[word] += freq
            
            # Add all prefixes of the word as subwords
            for i in range(1, len(word)):
                subword = word[:i]
                subword_vocab[subword] += freq

        return subword_vocab

    def encode(self, text):
        # Add [CLS] token at the start
        encoded_ids = [self.label2id['[CLS]']]
        
        # Tokenize input text into words
        words = text.split()
        
        for word in words:
            subwords = self.tokenize_word(word)
            encoded_ids.extend(subwords)

        # Add [SEP] token at the end
        encoded_ids.append(self.label2id['[SEP]'])
        
        return encoded_ids

    def tokenize_word(self, word):
        """Tokenizes a word into subwords according to the vocabulary."""
        subword_ids = []
        i = 0
        
        while i < len(word):
            found = False
            # Check for longest possible subword in the vocabulary
            for j in range(len(word), i, -1):
                subword = word[i:j]
                if subword in self.label2id:
                    if len(subword_ids) > 0:
                        subword_ids.append(self.label2id["##" + subword])  # Prefix subwords with '##'
                    else:
                        subword_ids.append(self.label2id[subword])
                    i = j
                    found = True
                    break
            if not found:  # If no match found, treat as [UNK]
                subword_ids.append(self.label2id['[UNK]'])
                break
        
        return subword_ids

    def decode(self, ids):
        # Reconstruct the original text from the encoded ids
        return " ".join(self.id2label.get(id, "[UNK]") for id in ids if id not in {self.label2id['[CLS]'], self.label2id['[SEP]']})

    def __len__(self):
        return len(self.label2id)

if __name__ == "__main__":
    file_path = "./text.txt"
    vocab = Vocab(file_path)
    print(f"Vocabulary size: {len(vocab)}")
    test_text = "hi I'm victor barra and I live in ghent"
    encoded = vocab.encode(test_text)
    print(f"Encoded: {encoded}")
    decoded = vocab.decode(encoded)
    print(f"Decoded: {decoded}")
