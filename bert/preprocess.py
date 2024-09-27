import os
from typing import List, Dict

class MovieDialogueProcessor:
    def __init__(self, conv_file: str, lines_file: str, max_len: int = 64):
        self.conv_file = conv_file
        self.lines_file = lines_file
        self.max_len = max_len
        self.lines_dic = {}
        self.pairs = []

    def load_data(self):
        """Load conversation and line data into memory."""
        with open(self.conv_file, 'r', encoding='iso-8859-1') as c:
            conv = c.readlines()
        with open(self.lines_file, 'r', encoding='iso-8859-1') as l:
            lines = l.readlines()

        self._split_lines(lines)
        self._generate_qa_pairs(conv)

    def _split_lines(self, lines: List[str]):
        """Split lines into a dictionary with line IDs as keys."""
        for line in lines:
            objects = line.split(" +++$+++ ")
            self.lines_dic[objects[0]] = objects[-1]

    def _generate_qa_pairs(self, conv: List[str]):
        """Generate question-answer pairs from conversations."""
        for con in conv:
            ids = eval(con.split(" +++$+++ ")[-1])
            for i in range(len(ids)):
                if i == len(ids) - 1:
                    break

                first = self.lines_dic[ids[i]].strip()
                second = self.lines_dic[ids[i + 1]].strip()

                # Add pairs, truncating to max_len
                qa_pair = [
                    ' '.join(first.split()[:self.max_len]),
                    ' '.join(second.split()[:self.max_len])
                ]
                self.pairs.append(qa_pair)

    def get_pairs(self) -> List[List[str]]:
        """Return the generated question-answer pairs."""
        return self.pairs


# Usage example
if __name__ == "__main__":
    corpus_movie_conv = './datasets/movie_conversations.txt'
    corpus_movie_lines = './datasets/movie_lines.txt'
    
    processor = MovieDialogueProcessor(corpus_movie_conv, corpus_movie_lines)
    processor.load_data()
    
    # Sample output
    print(processor.get_pairs())
