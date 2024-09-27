import os
from typing import List, Dict

class MovieDialogueProcessor:
    def __init__(self, conv_file: str, lines_file: str, max_len: int = 64):
        self.conv_file = conv_file
        self.lines_file = lines_file
        self.max_len = max_len
        self.lines_dic = {}  # Dictionary to hold line IDs and their corresponding text
        self.pairs = []      # List to hold question-answer pairs

    def load_data(self):
        """Load conversation and line data into memory."""
        with open(self.conv_file, 'r', encoding='iso-8859-1') as c:
            conv = c.readlines()
        with open(self.lines_file, 'r', encoding='iso-8859-1') as l:
            lines = l.readlines()

        # split lines into a dictionary
        self._split_lines(lines)
        # generate question-answer pairs from conversations
        self._generate_qa_pairs(conv)

    def _split_lines(self, lines: List[str]):
        """Split lines into a dictionary with line IDs as keys."""
        for line in lines:
            # split each line by the delimiter and store in the dictionary
            objects = line.split(" +++$+++ ")
            self.lines_dic[objects[0]] = objects[-1]  # map line ID to line text

    def _generate_qa_pairs(self, conv: List[str]):
        """Generate question-answer pairs from conversations."""
        for con in conv:
            # get line IDs from the conversation
            ids = eval(con.split(" +++$+++ ")[-1])
            for i in range(len(ids)):
                if i == len(ids) - 1:
                    break  # skip last ID as it has no following line

                # get corresponding lines for the current and next IDs
                first = self.lines_dic[ids[i]].strip()
                second = self.lines_dic[ids[i + 1]].strip()

                # add pairs, truncating to max_len
                qa_pair = [
                    ' '.join(first.split()[:self.max_len]),  # truncate first line
                    ' '.join(second.split()[:self.max_len])  # truncate second line
                ]
                self.pairs.append(qa_pair)

    def get_pairs(self) -> List[List[str]]:
        """Return the generated question-answer pairs."""
        return self.pairs 
    

if __name__ == "__main__":
    corpus_movie_conv = './datasets/movie_conversations.txt'
    corpus_movie_lines = './datasets/movie_lines.txt'
    
    processor = MovieDialogueProcessor(corpus_movie_conv, corpus_movie_lines)
    processor.load_data()
    
    # sample output question-answer pairs
    print(processor.get_pairs()[:20])
